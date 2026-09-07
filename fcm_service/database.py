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

supabase_client_var = ContextVar('supabase', default=None)
DEFAULT_TENANT_ID = (os.getenv("DEFAULT_TENANT_ID") or "skt").strip()
subdomain_var = ContextVar('subdomain', default=DEFAULT_TENANT_ID)

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

def fetch_device_token(user_id: str) -> Optional[str]:
    """
    특정 사용자의 FCM 디바이스 토큰을 조회합니다.
    
    Args:
        user_id (str): 사용자 ID (이메일)
        
    Returns:
        Optional[str]: 디바이스 토큰
    """
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        response = supabase.table('user_devices').select('device_token').eq('user_email', user_id).execute()
        
        if response.data:
            device_token = response.data[0].get('device_token')
            if device_token and device_token.strip():  # None이 아니고 빈 문자열이 아닌 경우
                return device_token
        
        return None
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        # 디바이스 토큰 조회
        device_token = fetch_device_token(user_id)
        if not device_token:
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
            response = messaging.send(message)
            success_count = 1
        except Exception as e:
            print(f"FCM 메시지 전송 오류: {e}")
            failed = True
        
        return {
            "success": success_count > 0,
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
        
        notification_data = {
            'title': notification_record.get('title', '새 알림'),
            'body': notification_record.get('description', '새로운 알림이 도착했습니다.'),
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
