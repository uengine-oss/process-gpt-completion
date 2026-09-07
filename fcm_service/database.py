import os
from supabase import create_client, Client
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import HTTPException
from datetime import datetime, timedelta
import pytz
from contextvars import ContextVar
from dotenv import load_dotenv
import socket
from firebase_admin import credentials, messaging
import firebase_admin
import logging
import asyncio

from recipients import (
    notification_text,
    resolve_user_emails as _resolve_user_emails,
    usable_tokens,
)

supabase_client_var = ContextVar('supabase', default=None)
subdomain_var = ContextVar('subdomain', default='localhost')

# 전역 변수로 변경
firebase_app = None

# Realtime 로그 설정
realtime_logger = logging.getLogger("realtime_subscriber")
if not realtime_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    realtime_logger.addHandler(handler)
    realtime_logger.setLevel(logging.INFO)

def setting_database():
    try:
        if os.getenv("ENV") != "production":
            load_dotenv()
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        supabase_client_var.set(supabase)
        
    except Exception as e:
        print(f"Database configuration error: {e}")

setting_database()

async def update_tenant_id(subdomain):
    try:
        if not subdomain:
            raise Exception("Unable to configure Tenant ID.")
        subdomain_var.set(subdomain)
    except Exception as e:
        print(f"An error occurred: {e}")

def _lookup_emails_by_uuid(uuids: List[str]) -> List[Dict[str, Any]]:
    """UUID 로 사용자 이메일을 찾는다. 규칙은 recipients 모듈에 있다."""
    supabase = supabase_client_var.get()
    if supabase is None:
        raise Exception("Supabase client is not configured for this request")
    response = supabase.table('users').select('id, email').in_('id', uuids).execute()
    return response.data or []


def resolve_user_emails(user_id: str) -> List[str]:
    """알림의 수신자 칸(이메일 · UUID · 콤마로 이은 여럿)을 이메일 목록으로 바꾼다."""
    return _resolve_user_emails(
        user_id,
        _lookup_emails_by_uuid,
        on_error=lambda e: realtime_logger.warning(f"사용자 UUID -> 이메일 변환 실패: {e}"),
    )


def fetch_device_tokens(user_id: str) -> List[str]:
    """
    수신자의 기기 토큰들을 조회한다.

    user_id 는 이메일일 수도, 사용자 UUID 일 수도, 콤마로 이어진 여럿일 수도
    있다 — 업무 알림은 UUID 로 오기 때문에 이 변환이 없으면 아무것도 못 찾는다.
    """
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")

        emails = resolve_user_emails(user_id)
        if not emails:
            return []

        response = supabase.table('user_devices').select('device_token').in_('user_email', emails).execute()
        return usable_tokens(response.data or [])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def fetch_device_token(user_id: str) -> Optional[str]:
    """
    첫 번째 기기 토큰. 기존 호출부(REST /device-token/{user_id})와의 호환을 위해 남깁니다.
    실제 발송은 fetch_device_tokens 로 전원에게 보냅니다.
    """
    tokens = fetch_device_tokens(user_id)
    return tokens[0] if tokens else None


def send_fcm_message(user_id: str, notification_data: dict) -> dict:
    """
    특정 사용자에게 FCM 푸시 알림을 전송합니다.
    
    Args:
        user_id (str): 사용자 ID (이메일)
        notification_data (dict): 알림 데이터
            - title: 알림 제목
            - body: 알림 내용
            - data: 추가적인 데이터 (dict)
            - type: 알림 타입 ('chat', 'workitem_bmp' 등)
        
    Returns:
        dict: 알림 전송 결과
    """
    try:
        global firebase_app
        # 기기 토큰 조회. 수신자가 여럿(콤마)일 수 있고, 이메일이 아니라
        # 사용자 UUID 로 올 수도 있다 — resolve_user_emails 가 둘 다 처리한다.
        device_tokens = fetch_device_tokens(user_id)
        if not device_tokens:
            return {"success": False, "message": "No device token found for the user"}
        
        # FCM 메시지 발송
        if not firebase_app:
            try:
                # Kubernetes 마운트된 시크릿에서 credentials 읽기
                secret_path = '/etc/secrets/firebase-credentials.json'
                if os.path.exists(secret_path):
                    cred = credentials.Certificate(secret_path)
                    firebase_app = firebase_admin.initialize_app(cred)
                else:
                    cred = credentials.Certificate('firebase-credentials.json')
                    firebase_app = firebase_admin.initialize_app(cred)
                
            except Exception as e:
                import traceback
                realtime_logger.error(f"Stack trace: {traceback.format_exc()}")
        
        if not firebase_app:
            raise Exception("Firebase app is not initialized")
        
        success_count = 0
        failed = False

        title = notification_data.get('title', '알림')
        body = notification_data.get('body', notification_data.get('description', ''))
        data = notification_data.get('data', {})
        data['type'] = notification_data.get('type', 'general')
        data['url'] = notification_data.get('url', '')
        sender_name = notification_data.get('from_user_id', '')  # 발신자 이름

        if sender_name:
            noti_title = sender_name
            noti_body = f"{body}\n{title}"
        else:
            noti_title = title
            noti_body = body

        data['title'] = noti_title
        data['body'] = noti_body

        # 담당자가 여럿인 업무는 기기도 여럿이다. 하나가 실패해도 나머지는 보낸다.
        for device_token in device_tokens:
            message = messaging.Message(
                token=device_token,
                notification=messaging.Notification(
                    title=noti_title,
                    body=noti_body
                ),
                data=data,
                android=messaging.AndroidConfig(
                    priority='high',
                ),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            badge=1,
                            sound='default'
                        )
                    )
                )
            )

            try:
                messaging.send(message)
                success_count += 1
            except Exception as e:
                print(f"FCM 메시지 전송 오류: {e}")
                failed = True

        return {
            "success": success_count > 0,
            "sent": success_count,
            "total": len(device_tokens),
            "message": "Message sent successfully" if success_count > 0 else "Failed to send message",
        }
    
    except Exception as e:
        print(f"FCM 메시지 전송 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))




def handle_new_notification(notification_record):
    """
    새로운 알림에 대해 FCM 푸시 알림을 전송하는 핸들러
    """
    try:
        
        user_id = notification_record.get('user_id')
        if not user_id:
            realtime_logger.warning("user_id가 없습니다.")
            return
        
        # FCM 알림 데이터 구성
        tenant_id = notification_record.get('tenant_id', '')
        url = notification_record.get('url', '')
        if tenant_id and url:
            url = f"https://{tenant_id}.process-gpt.io{url}"
        else:
            url = notification_record.get('url', '')

        print(f"url: {url}")
        
        # 본문에 인스턴스 식별자가 그대로 붙어 오는 것을 다듬는다.
        # 알림은 두 줄이 전부라, 절반이 UUID 면 무슨 일인지 알 수 없다.
        title, body = notification_text(
            notification_record.get('title'),
            notification_record.get('description'),
        )

        notification_data = {
            'title': title or '새 알림',
            'body': body or '새로운 알림이 도착했습니다.',
            'type': notification_record.get('type', 'general'),
            'url': url,
            'from_user_id': notification_record.get('from_user_id', ''),
            'data': {
                'notification_id': str(notification_record.get('id', '')),
                'url': notification_record.get('url', '')
            }
        }
        
        # FCM 메시지 전송
        result = send_fcm_message(user_id, notification_data)
        realtime_logger.info(f"FCM 알림 전송 결과: {result}")
        
    except Exception as e:
        realtime_logger.error(f"알림 처리 중 오류 발생: {e}")


def fetch_unprocessed_notifications() -> Optional[List[dict]]:
    try:
        pod_id = socket.gethostname()
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        env = os.getenv("ENV")

        # 1) ENV 기반 tenant 필터 적용 후 조회
        if env == 'dev':
            response = supabase.table('notifications') \
                .select('*') \
                .is_('consumer', 'null') \
                .eq('tenant_id', 'uengine') \
                .limit(10) \
                .execute()
        else:
            response = supabase.table('notifications') \
                .select('*') \
                .is_('consumer', 'null') \
                .neq('tenant_id', 'uengine') \
                .limit(10) \
                .execute()
        
        if not response.data:
            return None
        
        # 2) 배치 업데이트 시도
        notification_ids = [item['id'] for item in response.data]
        updated_notifications = []
        
        try:
            batch_update_response = supabase.table('notifications').update({
                'consumer': pod_id,
                'updated_at': datetime.now().isoformat()
            }).in_('id', notification_ids).is_('consumer', 'null').execute()
            
            if batch_update_response.data:
                updated_notifications = batch_update_response.data
                realtime_logger.info(f"Successfully claimed {len(updated_notifications)} notifications for pod {pod_id}")
            else:
                realtime_logger.info("No notifications were claimed in batch update")
                
        except Exception as batch_error:
            realtime_logger.warning(f"Batch update failed, falling back to individual updates: {batch_error}")
            
            # 3) 폴백: 개별 업데이트
            for notification in response.data:
                try:
                    update_response = supabase.table('notifications').update({
                        'consumer': pod_id,
                        'updated_at': datetime.now().isoformat()
                    }).eq('id', notification['id']).is_('consumer', 'null').execute()
                    
                    if update_response.data:
                        updated_notifications.append(update_response.data[0])
                        realtime_logger.info(f"Successfully claimed notification {notification['id']} for pod {pod_id}")
                    else:
                        realtime_logger.info(f"Notification {notification['id']} was already claimed by another pod")
                except Exception as e:
                    realtime_logger.warning(f"Failed to update notification {notification['id']}: {e}")
                    continue
        
        return updated_notifications if updated_notifications else None
        
    except Exception as e:
        realtime_logger.error(f"미처리 알림 fetch 실패: {str(e)}")
        return None


async def check_new_notifications():
    """
    미처리 알림을 체크하고 FCM 푸시를 전송합니다.
    """
    try:
        notifications = fetch_unprocessed_notifications()
        if notifications:
            
            for notification in notifications:
                handle_new_notification(notification)
        
    except Exception as e:
        realtime_logger.error(f"알림 체크 중 오류: {e}")


async def notification_polling_task():
    """
    15초마다 새로운 알림을 체크하는 폴링 태스크
    """
    while True:
        try:
            await check_new_notifications()
            await asyncio.sleep(15)  # 15초 대기
            
        except Exception as e:
            realtime_logger.error(f"폴링 태스크 오류: {e}")
            await asyncio.sleep(15)  # 오류 발생 시에도 15초 후 재시도