"""
Outbound 알림 서비스
workitem 생성 시 사용자의 prefer_contact 설정에 따라 전화 또는 SMS 발송
"""

import os
import uuid
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from database import supabase_client_var, fetch_user_info

if os.getenv("ENV") != "production":
    load_dotenv(override=True)

# Twilio 서버 URL (기본값: ngrok URL)
TWILIO_SERVER_URL = os.getenv("TWILIO_SERVER_URL", "https://monitor-faithful-slightly.ngrok-free.app")
TRIGGER_IDENTITY = "browser-user"


class OutboundNotifyError(Exception):
    """Outbound 알림 관련 에러"""
    pass


def insert_outbound_call_queue(
    workitem_id: str,
    user_id: str,
    tenant_id: str,
    channel: str,
    trigger_reason: str = "workitem_created"
) -> Optional[str]:
    """
    outbound_call_queue 테이블에 레코드 삽입
    
    Args:
        workitem_id: 워크아이템 ID
        user_id: 사용자 ID
        tenant_id: 테넌트 ID
        channel: 'phone' 또는 'sms'
        trigger_reason: 트리거 사유
    
    Returns:
        생성된 queue_id 또는 None (실패 시)
    """
    supabase = supabase_client_var.get()
    if supabase is None:
        raise OutboundNotifyError("Supabase client is not configured")
    
    queue_id = str(uuid.uuid4())
    queue_record = {
        "id": queue_id,
        "workitem_id": workitem_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "status": "PENDING",
        "channel": channel,
        "trigger_reason": trigger_reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    
    supabase.table("outbound_call_queue").insert(queue_record).execute()
    print(f"[OutboundNotify] Queue inserted: id={queue_id}, channel={channel}")
    return queue_id


def update_outbound_call_queue_status(
    queue_id: str,
    status: str,
    error_message: Optional[str] = None,
    **extra_fields
) -> None:
    """
    outbound_call_queue 상태 업데이트
    
    Args:
        queue_id: 큐 ID
        status: 새 상태 (PENDING, RINGING, CONNECTED, COMPLETED, FAILED 등)
        error_message: 에러 메시지 (실패 시)
        **extra_fields: 추가 필드 (started_at, connected_at, ended_at 등)
    """
    supabase = supabase_client_var.get()
    if supabase is None:
        raise OutboundNotifyError("Supabase client is not configured")
    
    update_data = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra_fields
    }
    
    if error_message:
        update_data["error_message"] = error_message
    
    supabase.table("outbound_call_queue").update(update_data).eq("id", queue_id).execute()
    print(f"[OutboundNotify] Queue {queue_id} updated to {status}")


def trigger_phone_call(queue_id: str, workitem_id: str) -> bool:
    """
    브라우저 전화 트리거
    
    Args:
        queue_id: 큐 ID
        workitem_id: 워크아이템 ID
    
    Returns:
        성공 여부
    """
    try:
        trigger_url = f"{TWILIO_SERVER_URL}/call/client"
        resp = requests.post(trigger_url, json={
            "identity": TRIGGER_IDENTITY,
            "workitem_id": workitem_id,
            "queue_id": queue_id
        }, timeout=5)
        resp.raise_for_status()
        
        # 상태 업데이트: RINGING
        update_outbound_call_queue_status(
            queue_id, 
            "RINGING",
            started_at=datetime.now(timezone.utc).isoformat()
        )
        return True
    except Exception as exc:
        # 상태 업데이트: FAILED
        update_outbound_call_queue_status(queue_id, "FAILED", error_message=str(exc))
        print(f"[OutboundNotify] Failed phone trigger: {exc}")
        return False


def send_sms(queue_id: str, phone_number: str, message: str) -> bool:
    """
    SMS 발송
    
    Args:
        queue_id: 큐 ID
        phone_number: 수신 전화번호 (E.164 형식)
        message: 메시지 내용
    
    Returns:
        성공 여부
    """
    try:
        sms_url = f"{TWILIO_SERVER_URL}/sms/outbound"
        resp = requests.post(sms_url, json={
            "to": phone_number,
            "body": message
        }, timeout=5)
        resp.raise_for_status()
        
        # SMS는 바로 COMPLETED (단방향)
        update_outbound_call_queue_status(
            queue_id,
            "COMPLETED",
            started_at=datetime.now(timezone.utc).isoformat(),
            ended_at=datetime.now(timezone.utc).isoformat()
        )
        print(f"[OutboundNotify] SMS sent to {phone_number}")
        return True
    except Exception as exc:
        # 상태 업데이트: FAILED
        update_outbound_call_queue_status(queue_id, "FAILED", error_message=str(exc))
        print(f"[OutboundNotify] Failed SMS: {exc}")
        return False


def process_outbound_notify(
    workitem_id: str,
    workitem_status: str,
    user_id: str,
    tenant_id: str,
    activity_name: str = "새 업무"
) -> None:
    """
    워크아이템에 대한 Outbound 알림 처리
    사용자의 prefer_contact 설정에 따라 전화 또는 SMS 발송
    
    Args:
        workitem_id: 워크아이템 ID
        workitem_status: 워크아이템 상태
        user_id: 담당자 ID
        tenant_id: 테넌트 ID
        activity_name: 활동 이름 (SMS 메시지에 사용)
    """
    print(f"[OutboundNotify] Processing: workitem={workitem_id}, status={workitem_status}, user={user_id}")
    
    # 상태 체크
    if workitem_status.upper() not in ("IN_PROGRESS", "TODO", "NEW"):
        print(f"[OutboundNotify] Skipped: status={workitem_status}")
        return
    
    # 사용자 정보 조회
    try:
        user_row = fetch_user_info(user_id)
    except Exception as e:
        print(f"[OutboundNotify] User lookup failed ({user_id}): {e}")
        return
    
    if not user_row:
        print(f"[OutboundNotify] Skipped: user not found")
        return
    
    prefer_contact = (user_row.get("prefer_contact") or "none").lower()
    phone_number = user_row.get("phone_number")
    
    print(f"[OutboundNotify] prefer_contact={prefer_contact}, phone_number={phone_number}")
    
    # prefer_contact가 none이거나 phone_number가 없으면 스킵
    if prefer_contact not in ("phone", "sms", "both"):
        print(f"[OutboundNotify] Skipped: prefer_contact={prefer_contact}")
        return
    
    if not phone_number:
        print(f"[OutboundNotify] Skipped: no phone_number")
        return
    
    # 채널 결정
    channel = "sms" if prefer_contact == "sms" else "phone"
    
    # Queue에 INSERT
    try:
        queue_id = insert_outbound_call_queue(
            workitem_id=workitem_id,
            user_id=user_id,
            tenant_id=tenant_id,
            channel=channel
        )
    except Exception as e:
        print(f"[OutboundNotify] Queue insert failed: {e}")
        return
    
    # 실제 발송
    if channel == "phone":
        trigger_phone_call(queue_id, workitem_id)
    elif channel == "sms":
        sms_body = f"[ProcessGPT] 새 업무가 도착했습니다: {activity_name}"
        send_sms(queue_id, phone_number, sms_body)
