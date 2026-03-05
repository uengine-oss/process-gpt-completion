#!/usr/bin/env python3
"""
프로세스 정의 마이그레이션 스크립트
목적: JSON definition의 activity 메타데이터를 BPMN XML로 동기화
- 기존 액티비티의 properties 키에서 checkpoints, description, instruction 값을 추출하여 새로운 구조로 병합
- Lock 테이블 조건을 적용하여 안전한 마이그레이션 지원

사용법:
1. 환경변수 설정 (.env 파일 또는 시스템 환경변수):
   - SUPABASE_URL: Supabase 프로젝트 URL
   - SUPABASE_KEY: Supabase 서비스 키

2. 데이터베이스 백업 테이블 생성 및 함수 설정:
   - migration_rpc_function.sql 파일의 함수를 데이터베이스에 실행하세요

3. 실행:
   - 시뮬레이션 모드: python migration_script.py --dry-run
   - 실제 실행: python migration_script.py
   - Lock 조건 적용: python migration_script.py --lock-user-id "user123"
   - 특정 테넌트만 처리: python migration_script.py --tenant-id "tenant123"
   - 배치 크기 지정: python migration_script.py --batch-size 10
   - 최대 배치 수 지정: python migration_script.py --max-batches 10

주의사항:
- 실행 전 반드시 --dry-run으로 테스트해보세요
- 백업 테이블(proc_def_backup)이 미리 생성되어 있어야 합니다
- 마이그레이션 전에 대상 프로세스들이 자동으로 백업됩니다
- 특정 테넌트만 처리하려면 --tenant-id 옵션을 사용하세요
- Lock 조건을 적용하려면 --lock-user-id 옵션을 사용하세요 (lock이 없거나 해당 user_id인 경우만 마이그레이션)
"""

import os
import argparse
from supabase import create_client, Client
import json
import defusedxml.ElementTree as ET
from datetime import datetime
import sys
import logging
from dotenv import load_dotenv
from contextvars import ContextVar

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supabase 클라이언트 전역 변수
supabase_client_var = ContextVar('supabase', default=None)


class ActivityMetadataMigrator:
    """액티비티 메타데이터 마이그레이션 클래스"""
    
    # XML 네임스페이스
    NAMESPACES = {
        'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
        'uengine': 'http://uengine'
    }
    
    def __init__(self):
        """Supabase 클라이언트를 사용한 마이그레이션 클래스 초기화"""
        self.supabase = None
        
    def setup_supabase(self):
        """Supabase 클라이언트 설정"""
        try:
            load_dotenv(override=True)

            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")
            
            if not supabase_url or not supabase_key:
                raise Exception("SUPABASE_URL과 SUPABASE_KEY 환경변수가 설정되지 않았습니다.")
            
            self.supabase = create_client(supabase_url, supabase_key)
            supabase_client_var.set(self.supabase)
            logger.info("Supabase 클라이언트 연결 성공")
        except Exception as e:
            logger.error(f"Supabase 클라이언트 연결 실패: {e}")
            raise
    
    def backup_target_processes(self, processes, tenant_id: str | None = None):
        """마이그레이션 대상 프로세스들을 백업 테이블에 저장"""
        try:
            if not processes:
                logger.info("백업할 프로세스가 없습니다.")
                return
            
            # 백업 테이블에 대상 프로세스들 삽입
            for proc_id, proc_name, definition, bpmn in processes:
                backup_data = {
                    'id': proc_id,
                    'name': proc_name,
                    'definition': definition,
                    'bpmn': bpmn
                }
                if tenant_id:
                    backup_data['tenant_id'] = tenant_id
                
                # 기존 데이터가 있으면 삭제 후 삽입 (UPSERT 방식)
                delete_query = self.supabase.table('proc_def_backup').delete().eq('id', proc_id)
                if tenant_id:
                    delete_query = delete_query.eq('tenant_id', tenant_id)
                delete_query.execute()
                response = self.supabase.table('proc_def_backup').insert(backup_data).execute()
                
                if not response.data:
                    logger.warning(f"백업 실패: {proc_id} ({proc_name})")
                else:
                    logger.info(f"백업 완료: {proc_id} ({proc_name})")
            
            logger.info(f"총 {len(processes)}개 프로세스 백업 완료")
            
        except Exception as e:
            logger.error(f"백업 과정에서 오류 발생: {e}")
            raise
    
    def get_target_processes(self, batch_size: int = 5, cursor_after_id: str | None = None, tenant_id: str | None = None, lock_user_id: str | None = None):
        """마이그레이션 대상 프로세스 조회(배치)
        Args:
            batch_size: 한 번에 가져올 최대 프로세스 수
            cursor_after_id: 이 ID보다 큰 레코드(id 기준)만 조회하는 커서
            tenant_id: 특정 테넌트만 대상으로 제한
            lock_user_id: lock 테이블에서 허용할 user_id (lock이 없거나 이 user_id인 경우만 마이그레이션)
        """
        try:
            # lock_user_id가 설정된 경우 lock 테이블 조건을 적용
            if lock_user_id:
                # PostgREST의 RPC 기능을 사용하여 복잡한 쿼리 실행
                # lock 테이블에 id가 없거나 user_id가 지정된 값인 경우만 조회
                rpc_params = {
                    'batch_size': batch_size,
                    'cursor_after_id': cursor_after_id,
                    'target_tenant_id': tenant_id,
                    'lock_user_id': lock_user_id
                }
                
                # RPC 함수 호출 (이 함수는 데이터베이스에서 미리 정의되어야 함)
                response = self.supabase.rpc('get_migration_target_processes', rpc_params).execute()
                
                if not response.data:
                    return []
                
                # RPC 응답에서 필요한 필드 추출
                results = []
                for row in response.data:
                    try:
                        # definition이 문자열이면 JSON 파싱, 이미 딕셔너리면 그대로 사용
                        if isinstance(row['definition'], str):
                            try:
                                definition = json.loads(row['definition'])
                            except json.JSONDecodeError:
                                logger.warning(f"definition JSON 파싱 실패: {row['id']}")
                                continue
                        elif isinstance(row['definition'], dict):
                            definition = row['definition']
                        else:
                            continue
                        
                        if 'activities' in definition:
                            results.append((row['id'], row['name'], row['definition'], row['bpmn']))
                    except (TypeError, AttributeError) as e:
                        logger.warning(f"definition 처리 오류 {row['id']}: {e}")
                        continue
                
                logger.info(f"조회된 배치 대상 (lock 조건 적용): {len(results)}개 프로세스")
                return results
            
            else:
                # 기존 로직 (lock 조건 없음)
                query = self.supabase.table('proc_def').select('id, name, definition, bpmn').filter(
                    'isdeleted', 'eq', False
                ).filter(
                    'definition', 'not.is', 'null'
                ).filter(
                    'bpmn', 'not.is', 'null'
                ).or_(
                    'bpmn.like.%"variableForHtmlFormContext"%,bpmn.like.%"inputMapping"%,bpmn.like.%"outputMapping"%'
                ).order('id')

                if tenant_id:
                    query = query.eq('tenant_id', tenant_id)

                if cursor_after_id:
                    query = query.gt('id', cursor_after_id)

                response = query.limit(batch_size).execute()
                
                # JSON 필터링은 클라이언트 사이드에서 처리
                results = []
                for row in response.data:
                    try:
                        # definition이 문자열이면 JSON 파싱, 이미 딕셔너리면 그대로 사용
                        if isinstance(row['definition'], str):
                            try:
                                definition = json.loads(row['definition'])
                            except json.JSONDecodeError:
                                logger.warning(f"definition JSON 파싱 실패: {row['id']}")
                                continue
                        elif isinstance(row['definition'], dict):
                            definition = row['definition']
                        else:
                            continue
                        
                        if 'activities' in definition:
                            results.append((row['id'], row['name'], row['definition'], row['bpmn']))
                    except (TypeError, AttributeError) as e:
                        logger.warning(f"definition 처리 오류 {row['id']}: {e}")
                        continue
                
                logger.info(f"조회된 배치 대상: {len(results)}개 프로세스")
                return results
        except Exception as e:
            logger.error(f"프로세스 조회 실패: {e}")
            raise
    
    def build_activity_properties(self, activity_json):
        """액티비티 JSON에서 새로운 properties 구성"""
        # 기본 properties 구성
        new_properties = {
            'role': activity_json.get('role', ''),
            'duration': activity_json.get('duration', 5),
            'instruction': activity_json.get('instruction', ''),
            'description': activity_json.get('description', ''),
            'checkpoints': activity_json.get('checkpoints', []),
            'agentMode': activity_json.get('agentMode', 'none'),
            'orchestration': activity_json.get('orchestration', 'none'),
            'attachments': activity_json.get('attachments', []),
            'inputData': activity_json.get('inputData', []),
            'tool': activity_json.get('tool', '')
        }
        
        # 기존 properties에서 값 추출하여 병합
        existing_properties = self.parse_existing_properties(activity_json.get('properties'))
        if existing_properties:
            # 기존 properties의 값이 있으면 우선 적용
            for key in ['checkpoints', 'description', 'instruction']:
                if key in existing_properties and existing_properties[key]:
                    new_properties[key] = existing_properties[key]
        
        return new_properties
    
    def parse_existing_properties(self, properties_str):
        """기존 properties 문자열에서 JSON 파싱하여 필요한 값들 추출"""
        if not properties_str:
            return None
            
        try:
            # properties가 문자열인 경우 JSON 파싱
            if isinstance(properties_str, str):
                existing_props = json.loads(properties_str)
            elif isinstance(properties_str, dict):
                existing_props = properties_str
            else:
                return None
            
            # 필요한 키들만 추출
            result = {}
            for key in ['checkpoints', 'description', 'instruction']:
                if key in existing_props:
                    value = existing_props[key]
                    # 빈 값이 아닌 경우만 포함
                    if value is not None and value != '' and value != []:
                        result[key] = value
            
            return result if result else None
            
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            logger.warning(f"기존 properties 파싱 실패: {e}")
            return None
    
    def update_xml_activity(self, xml_string, activity_id, new_properties):
        """XML에서 특정 액티비티의 uengine:json 업데이트"""
        try:
            # XML 네임스페이스 등록
            for prefix, uri in self.NAMESPACES.items():
                ET.register_namespace(prefix, uri)
            
            # XML 파싱
            root = ET.fromstring(xml_string)
            
            activity_types = ['userTask', 'serviceTask', 'sendTask', 'receiveTask', 'scriptTask', 'manualTask']
            activity_element = None
            
            for activity_type in activity_types:
                xpath = f".//{{{self.NAMESPACES['bpmn']}}}{activity_type}[@id='{activity_id}']"
                activity_element = root.find(xpath)
                if activity_element is not None:
                    break
            
            if activity_element is None:
                logger.warning(f"액티비티를 찾을 수 없음: {activity_id}")
                return xml_string
            
            # extensionElements 찾기 또는 생성
            ext_elem = activity_element.find(f"{{{self.NAMESPACES['bpmn']}}}extensionElements")
            if ext_elem is None:
                ext_elem = ET.SubElement(activity_element, f"{{{self.NAMESPACES['bpmn']}}}extensionElements")
            
            # uengine:properties 찾기 또는 생성
            props_elem = ext_elem.find(f"{{{self.NAMESPACES['uengine']}}}properties")
            if props_elem is None:
                props_elem = ET.SubElement(ext_elem, f"{{{self.NAMESPACES['uengine']}}}properties")
            
            # uengine:json 찾기 또는 생성
            json_elem = props_elem.find(f"{{{self.NAMESPACES['uengine']}}}json")
            if json_elem is None:
                json_elem = ET.SubElement(props_elem, f"{{{self.NAMESPACES['uengine']}}}json")
            
            # JSON 데이터 업데이트
            json_elem.text = json.dumps(new_properties, ensure_ascii=False, separators=(',', ':'))
            
            # XML을 문자열로 변환 (XML 선언 포함)
            xml_string = ET.tostring(root, encoding='unicode', method='xml', xml_declaration=True)
            
            return xml_string
            
        except Exception as e:
            logger.error(f"XML 업데이트 실패 (activity: {activity_id}): {e}")
            raise
    
    def migrate_process(self, proc_id, proc_name, definition_json, bpmn_xml):
        """단일 프로세스 마이그레이션"""
        try:
            # definition이 문자열이면 JSON 파싱, 이미 딕셔너리면 그대로 사용
            if isinstance(definition_json, str):
                try:
                    definition = json.loads(definition_json)
                except json.JSONDecodeError as e:
                    logger.error(f"  {proc_id}: definition JSON 파싱 실패 - {e}")
                    return -1, None, None
            elif isinstance(definition_json, dict):
                definition = definition_json
            else:
                logger.error(f"  {proc_id}: definition 데이터 타입 오류 - {type(definition_json)}")
                return -1, None, None
            
            activities = definition.get('activities', [])
            
            if not activities:
                logger.info(f"  {proc_id}: 액티비티 없음, 건너뜀")
                return 0, None, None
            
            updated_xml = bpmn_xml
            updated_definition = definition.copy()
            updated_count = 0
            
            # 각 액티비티 처리
            for i, activity in enumerate(activities):
                activity_id = activity.get('id')
                activity_type = activity.get('type')
                
                if activity_type not in ['userTask', 'serviceTask', 'sendTask', 'receiveTask', 'scriptTask', 'manualTask']:
                    continue
                
                # 새로운 properties 구성 (기존 properties에서 값 병합)
                new_properties = self.build_activity_properties(activity)
                
                # 기존 properties에서 병합된 값이 있는지 로그 출력
                existing_properties = self.parse_existing_properties(activity.get('properties'))
                if existing_properties:
                    merged_keys = []
                    for key in ['checkpoints', 'description', 'instruction']:
                        if key in existing_properties and existing_properties[key]:
                            merged_keys.append(key)
                    if merged_keys:
                        logger.info(f"  {proc_id} - {activity_id}: 기존 properties에서 병합된 키: {', '.join(merged_keys)}")
                
                # XML 업데이트
                updated_xml = self.update_xml_activity(updated_xml, activity_id, new_properties)
                
                # Definition JSON의 액티비티 업데이트
                updated_activity = activity.copy()
                
                # 기존 properties와 outputData 키 제거 (중요!)
                if 'properties' in updated_activity:
                    del updated_activity['properties']
                if 'outputData' in updated_activity:
                    del updated_activity['outputData']
                
                # 새로운 properties 추가
                updated_activity.update(new_properties)
                updated_definition['activities'][i] = updated_activity
                
                updated_count += 1
            
            if updated_count > 0:
                logger.info(f"  {proc_id} ({proc_name}): {updated_count}개 액티비티 업데이트")
                return updated_count, updated_xml, updated_definition
            else:
                return 0, None, None
                
        except Exception as e:
            logger.error(f"  {proc_id}: 마이그레이션 실패 - {e}")
            return -1, None, None
    
    def save_migrated_process(self, proc_id, updated_xml, updated_definition, tenant_id: str | None = None):
        """마이그레이션된 프로세스 저장"""
        try:
            # JSONB 타입에 맞게 딕셔너리를 그대로 저장
            update_data = {
                'bpmn': updated_xml,
                'definition': updated_definition  # 딕셔너리 그대로 저장 (Supabase가 JSONB로 자동 변환)
            }
            
            update_query = self.supabase.table('proc_def').update(update_data).eq('id', proc_id)
            if tenant_id:
                update_query = update_query.eq('tenant_id', tenant_id)
            response = update_query.execute()
            
            if not response.data:
                raise Exception(f"프로세스 {proc_id}를 찾을 수 없습니다.")
                
        except Exception as e:
            logger.error(f"  {proc_id}: 저장 실패 - {e}")
            raise
    
    def run_migration(self, dry_run=False, batch_size: int = 5, max_batches: int = None, tenant_id: str | None = None, lock_user_id: str | None = None):
        """마이그레이션 실행
        
        Args:
            dry_run (bool): True면 실제 업데이트 없이 시뮬레이션만 수행
            batch_size (int): 배치 크기(기본 5)
            max_batches (int|None): 최대 배치 수(미설정 시 끝까지)
            tenant_id (str|None): 특정 테넌트만 처리
            lock_user_id (str|None): lock 테이블에서 허용할 user_id (lock이 없거나 이 user_id인 경우만 마이그레이션)
        """
        logger.info("=" * 70)
        logger.info("프로세스 정의 마이그레이션 시작")
        logger.info(f"모드: {'DRY RUN (시뮬레이션)' if dry_run else 'LIVE (실제 업데이트)'}")
        logger.info(f"배치 크기: {batch_size}")
        if max_batches is not None:
            logger.info(f"최대 배치 수: {max_batches}")
        if tenant_id:
            logger.info(f"테넌트: {tenant_id}")
        if lock_user_id:
            logger.info(f"Lock 허용 user_id: {lock_user_id}")
        logger.info("=" * 70)
        
        try:
            self.setup_supabase()
            
            success_count = 0
            fail_count = 0
            total_activities = 0
            total_processes = 0

            batch_index = 0
            cursor_id = None

            while True:
                if max_batches is not None and batch_index >= max_batches:
                    logger.info(f"최대 배치 수({max_batches})에 도달하여 중단합니다.")
                    break

                processes = self.get_target_processes(batch_size=batch_size, cursor_after_id=cursor_id, tenant_id=tenant_id, lock_user_id=lock_user_id)

                if not processes:
                    if batch_index == 0:
                        logger.info("마이그레이션 대상이 없습니다.")
                    break

                # 마이그레이션 전 백업(현재 배치만)
                if not dry_run:
                    logger.info("\n현재 배치 백업 중...")
                    self.backup_target_processes(processes, tenant_id=tenant_id)
                    logger.info("백업 완료\n")

                logger.info(f"배치 {batch_index + 1} 처리 시작 (건수: {len(processes)})")
                
                for proc_id, proc_name, definition, bpmn in processes:
                    updated_count, updated_xml, updated_definition = self.migrate_process(
                        proc_id, proc_name, definition, bpmn
                    )
                    
                    if updated_count > 0:
                        if not dry_run:
                            self.save_migrated_process(proc_id, updated_xml, updated_definition, tenant_id=tenant_id)
                        success_count += 1
                        total_activities += updated_count
                    elif updated_count < 0:
                        fail_count += 1
                
                total_processes += len(processes)
                batch_index += 1

                # 다음 배치를 위한 커서 갱신(마지막 id)
                cursor_id = processes[-1][0]
            
            # 결과 요약
            logger.info("\n" + "=" * 70)
            logger.info("마이그레이션 완료")
            logger.info(f"완료된 프로세스: {', '.join([proc_name for _, proc_name, _, _ in processes])}")
            logger.info(f"총 프로세스: {total_processes}")
            logger.info(f"성공: {success_count}")
            logger.info(f"실패: {fail_count}")
            logger.info(f"업데이트된 액티비티 총 개수: {total_activities}")
            
            if not dry_run and success_count > 0:
                logger.info("\n" + "=" * 50)
                logger.info("백업 및 롤백 정보")
                logger.info("=" * 50)
                logger.info("백업 테이블: proc_def_backup")
                logger.info("문제 발생 시 다음 쿼리로 롤백 가능:")
                logger.info("UPDATE proc_def p SET bpmn = b.bpmn, definition = b.definition FROM proc_def_backup b WHERE p.id = b.id;")
                logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"\n마이그레이션 실패: {e}")
            raise


def main():
    """메인 함수"""
    
    # CLI 인자 파싱
    parser = argparse.ArgumentParser(description='Process Definition Migration')
    parser.add_argument('--dry-run', action='store_true', help='시뮬레이션 모드로 실행 (DB 업데이트 없음)')
    parser.add_argument('--batch-size', type=int, default=5, help='배치 크기 (기본 5)')
    parser.add_argument('--max-batches', type=int, default=None, help='최대 배치 수 (미설정 시 전체 처리)')
    parser.add_argument('--tenant-id', type=str, default=None, help='특정 테넌트만 처리')
    parser.add_argument('--lock-user-id', type=str, default=None, help='lock 테이블에서 허용할 user_id (lock이 없거나 이 user_id인 경우만 마이그레이션)')
    args = parser.parse_args()
    
    try:
        migrator = ActivityMetadataMigrator()
        migrator.run_migration(
            dry_run=args.dry_run,
            batch_size=max(1, args.batch_size),
            max_batches=args.max_batches,
            tenant_id=args.tenant_id,
            lock_user_id=args.lock_user_id
        )
        
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


"""
사용 예시:
- 시뮬레이션 모드로 실행
uv run migration_script.py --dry-run --batch-size 1 --max-batches 1 --tenant-id uengine --lock-user-id "오순영"
- 실제 실행
uv run migration_script.py --tenant-id localhost
uv run migration_script.py --batch-size 3 --max-batches 5 --tenant-id localhost
uv run migration_script.py --batch-size 1 --max-batches 5 --tenant-id uengine --lock-user-id "오순영"
"""