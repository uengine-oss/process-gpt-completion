"""
문서 파일 파싱 및 요약 유틸리티
Upstage AI Document Parser와 LangChain Summarization 사용
"""
import os
import tempfile
import httpx
from typing import Optional, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from llm_factory import create_llm
import logging

logger = logging.getLogger(__name__)

# 지원하는 파일 확장자
SUPPORTED_EXTENSIONS = {'.pdf', '.xlsx', '.xls', '.xlsm', '.docx', '.doc', '.hwp', '.hwpx', '.pptx', '.ppt'}

# 요약이 필요한 최소 문자 수 (약 5000자 이상이면 요약)
SUMMARIZATION_THRESHOLD = 5000

# Upstage API 설정
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
UPSTAGE_DOCUMENT_PARSE_URL = "https://api.upstage.ai/v1/document-digitization"
UPSTAGE_MODEL = "document-parse"


async def parse_document_with_upstage(file_path: str, file_url: Optional[str] = None) -> Optional[str]:
    """
    Upstage AI Document Parser를 사용하여 문서 파싱
    
    Args:
        file_path: 로컬 파일 경로 (우선순위)
        file_url: 파일 URL (file_path가 없을 때 사용)
    
    Returns:
        파싱된 텍스트 또는 None
    """
    if not UPSTAGE_API_KEY:
        logger.warning("[WARNING] UPSTAGE_API_KEY가 설정되지 않았습니다.")
        return None
    
    try:
        headers = {
            "Authorization": f"Bearer {UPSTAGE_API_KEY}"
        }
        
        # file_path가 URL인지 확인
        is_url = file_path and (file_path.startswith('http://') or file_path.startswith('https://'))
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            if file_path and not is_url and os.path.exists(file_path):
                # 로컬 파일 업로드
                logger.info(f"[INFO] 로컬 파일 업로드: {file_path}")
                with open(file_path, 'rb') as f:
                    files = {'document': (os.path.basename(file_path), f, 'application/octet-stream')}
                    data = {
                        'ocr': 'force',
                        'model': UPSTAGE_MODEL
                    }
                    response = await client.post(
                        UPSTAGE_DOCUMENT_PARSE_URL,
                        headers=headers,
                        files=files,
                        data=data
                    )
            elif is_url or file_url:
                # URL로 파싱 (file_path가 URL이거나 file_url이 있는 경우)
                target_url = file_path if is_url else file_url
                logger.info(f"[INFO] URL로 파싱: {target_url}")
                
                # URL 파일 다운로드 후 업로드
                file_response = await client.get(target_url)
                if file_response.status_code == 200:
                    file_name = os.path.basename(target_url.split('?')[0])
                    files = {'document': (file_name, file_response.content, 'application/octet-stream')}
                    data = {
                        'ocr': 'force',
                        'model': UPSTAGE_MODEL
                    }
                    response = await client.post(
                        UPSTAGE_DOCUMENT_PARSE_URL,
                        headers=headers,
                        files=files,
                        data=data
                    )
                else:
                    logger.error(f"[ERROR] URL 파일 다운로드 실패: {file_response.status_code}")
                    return None
            else:
                logger.error("[ERROR] 파일 경로 또는 URL이 제공되지 않았습니다.")
                return None
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[DEBUG] Upstage API 응답: {result.keys() if isinstance(result, dict) else type(result)}")
                
                # Upstage Document Digitization API 응답 형식
                text_parts = []
                
                # 1. content.html에서 HTML 태그 제거하고 텍스트 추출
                if 'content' in result:
                    content = result.get('content', {})
                    if isinstance(content, dict):
                        # HTML이 있으면 파싱
                        if 'html' in content:
                            import re
                            html_text = content['html']
                            # HTML 태그 제거
                            text_without_tags = re.sub(r'<[^>]+>', ' ', html_text)
                            # 연속된 공백 및 줄바꿈 정리
                            text_without_tags = re.sub(r'\s+', ' ', text_without_tags)
                            text_parts.append(text_without_tags.strip())
                        # text가 있으면 직접 사용
                        elif 'text' in content:
                            text_parts.append(content['text'])
                    elif isinstance(content, str):
                        text_parts.append(content)
                
                # 2. elements 배열에서 텍스트 추출 (보조)
                if not text_parts and 'elements' in result:
                    elements = result.get('elements', [])
                    for element in elements:
                        if 'text' in element:
                            text_parts.append(element['text'])
                
                # 3. pages 배열에서 텍스트 추출 (다른 API 버전)
                if not text_parts and 'pages' in result:
                    pages = result.get('pages', [])
                    for page in pages:
                        if 'text' in page:
                            text_parts.append(page['text'])
                        elif 'elements' in page:
                            for element in page['elements']:
                                if 'text' in element:
                                    text_parts.append(element['text'])
                
                text = '\n'.join(text_parts)
                
                if text:
                    logger.info(f"[INFO] 문서 파싱 성공: {len(text)} 문자")
                    return text
                else:
                    logger.warning(f"[WARNING] 파싱된 텍스트가 없습니다. 응답 키: {list(result.keys())}")
                    return None
            else:
                logger.error(f"[ERROR] Upstage API 오류: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"[ERROR] 문서 파싱 실패: {str(e)}")
        return None


async def summarize_text(text: str, max_length: int = 2000) -> str:
    """
    LangChain을 사용하여 긴 텍스트 요약
    
    Args:
        text: 요약할 텍스트
        max_length: 목표 최대 길이
    
    Returns:
        요약된 텍스트
    """
    try:
        # 텍스트가 임계값보다 짧으면 그대로 반환
        if len(text) <= SUMMARIZATION_THRESHOLD:
            logger.info(f"[INFO] 텍스트가 짧아 요약 불필요: {len(text)} 문자")
            return text
        
        logger.info(f"[INFO] 텍스트 요약 시작: {len(text)} 문자")
        
        # LLM 생성
        llm = create_llm(streaming=False, temperature=0.3)
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Document 객체로 변환
        docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]
        
        # 요약 체인 생성 (map_reduce 방식)
        summarize_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            verbose=False
        )
        
        # 요약 실행
        summary = await summarize_chain.ainvoke({"input_documents": docs})
        summarized_text = summary.get('output_text', '')
        
        # 여전히 너무 길면 추가 요약
        if len(summarized_text) > max_length * 1.5:
            logger.info(f"[INFO] 추가 요약 수행: {len(summarized_text)} 문자")
            refine_chain = load_summarize_chain(
                llm=llm,
                chain_type="stuff",
                verbose=False
            )
            final_summary = await refine_chain.ainvoke({
                "input_documents": [Document(page_content=summarized_text)]
            })
            summarized_text = final_summary.get('output_text', '')
        
        logger.info(f"[INFO] 텍스트 요약 완료: {len(text)} -> {len(summarized_text)} 문자")
        return summarized_text
        
    except Exception as e:
        logger.error(f"[ERROR] 텍스트 요약 실패: {str(e)}")
        # 요약 실패시 원본의 일부만 반환
        return text[:max_length] + "... (요약 실패)"


async def process_document_file(
    file_path: Optional[str] = None,
    file_url: Optional[str] = None,
    file_name: Optional[str] = None
) -> Optional[str]:
    """
    문서 파일을 파싱하고 필요시 요약하여 텍스트 반환
    
    Args:
        file_path: 로컬 파일 경로 또는 URL (Supabase storage URL 포함)
        file_url: 파일 URL (file_path가 로컬 경로일 때만 사용)
        file_name: 파일 이름 (확장자 확인용)
    
    Returns:
        처리된 텍스트 또는 None
    """
    try:
        # file_path가 URL인지 확인
        is_file_path_url = file_path and (file_path.startswith('http://') or file_path.startswith('https://'))
        
        # 파일 확장자 확인
        if file_name:
            ext = os.path.splitext(file_name.lower())[1]
        elif file_path:
            # URL에서 확장자 추출 (쿼리 파라미터 제거)
            if is_file_path_url:
                path_part = file_path.split('?')[0]
                ext = os.path.splitext(path_part.lower())[1]
            else:
                ext = os.path.splitext(file_path.lower())[1]
        elif file_url:
            ext = os.path.splitext(file_url.lower())[1].split('?')[0]  # URL 쿼리 파라미터 제거
        else:
            logger.warning("[WARNING] 파일 정보가 제공되지 않았습니다.")
            return None
        
        # 지원하는 확장자인지 확인
        if ext not in SUPPORTED_EXTENSIONS:
            logger.info(f"[INFO] 지원하지 않는 파일 형식: {ext}")
            return None
        
        display_name = file_name or file_path or file_url
        logger.info(f"[INFO] 문서 파일 처리 시작: {display_name}")
        logger.info(f"[INFO] 파일 타입: {'URL' if is_file_path_url else '로컬 경로'}, 확장자: {ext}")
        
        # 1. Upstage로 문서 파싱
        parsed_text = await parse_document_with_upstage(file_path, file_url)
        
        if not parsed_text:
            logger.warning("[WARNING] 문서 파싱 결과가 없습니다.")
            return None
        
        # 2. 필요시 요약
        processed_text = await summarize_text(parsed_text)
        
        return processed_text
        
    except Exception as e:
        logger.error(f"[ERROR] 문서 파일 처리 실패: {str(e)}")
        return None


def is_document_file(field_value: Any) -> bool:
    """
    필드 값이 문서 파일인지 확인
    
    Args:
        field_value: 필드 값 (dict, str 등)
    
    Returns:
        문서 파일 여부
    """
    try:
        # dict 형태의 파일 정보
        if isinstance(field_value, dict):
            file_name = field_value.get('name') or field_value.get('fileName') or field_value.get('file_name')
            if file_name:
                ext = os.path.splitext(file_name.lower())[1]
                return ext in SUPPORTED_EXTENSIONS
        
        # URL 문자열
        elif isinstance(field_value, str):
            if field_value.startswith('http') or field_value.startswith('/'):
                ext = os.path.splitext(field_value.lower())[1].split('?')[0]
                return ext in SUPPORTED_EXTENSIONS
        
        return False
        
    except Exception:
        return False


async def extract_file_info(field_value: Any) -> Optional[Dict[str, str]]:
    """
    필드 값에서 파일 정보 추출
    
    Args:
        field_value: 필드 값
    
    Returns:
        파일 정보 dict (file_name, file_path, file_url)
    """
    try:
        if isinstance(field_value, dict):
            # dict에서 파일 정보 추출
            file_name = field_value.get('name') or field_value.get('fileName') or field_value.get('file_name')
            file_path = field_value.get('path') or field_value.get('filePath')
            file_url = field_value.get('url') or field_value.get('fileUrl') or field_value.get('file_url')
            
            # path가 URL인 경우 file_path로 사용 (Supabase storage URL 등)
            if file_path and (file_path.startswith('http://') or file_path.startswith('https://')):
                logger.info(f"[INFO] path 필드가 URL입니다: {file_path}")
                return {
                    'file_name': file_name or os.path.basename(file_path.split('?')[0]),
                    'file_path': file_path,  # URL을 file_path로 전달
                    'file_url': file_url
                }
            
            return {
                'file_name': file_name,
                'file_path': file_path,
                'file_url': file_url
            }
        elif isinstance(field_value, str):
            # 문자열인 경우
            is_url = field_value.startswith('http://') or field_value.startswith('https://')
            return {
                'file_name': os.path.basename(field_value.split('?')[0]),
                'file_path': field_value if is_url else field_value,  # URL도 file_path로 전달
                'file_url': field_value if is_url else None
            }
        
        return None
        
    except Exception as e:
        logger.error(f"[ERROR] 파일 정보 추출 실패: {str(e)}")
        return None

