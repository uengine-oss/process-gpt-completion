import asyncio
import signal
from typing import Set
import os
import socket

from database import (
    setting_database, fetch_workitem_with_submitted_status, 
    upsert_workitem, cleanup_stale_consumers,
    fetch_process_definition_by_version, fetch_workitem_with_pending_status,
    fetch_process_instance, upsert_process_instance
)
from workitem_processor import handle_workitem, handle_service_workitem, handle_pending_workitem
from file_cleanup_service import file_cleanup_polling_task
CONSUMER_FILTER = os.getenv("WORKITEM_CONSUMER")  # ?? "worker-a"

# ?꾩뿭 蹂?섎줈 ?꾩옱 ?ㅽ뻾 以묒씤 ?쒖뒪?щ뱾??異붿쟻
running_tasks: Set[asyncio.Task] = set()
shutdown_event = asyncio.Event()

async def safe_handle_workitem(workitem):
    try:
        # consumer ?쒖쇅 洹쒖튃: consumer媛 "CONSUMER_FILTER? pod_id瑜?紐⑤몢 ?ы븿"?섎㈃ ?ㅽ궢
        if CONSUMER_FILTER:
            current_consumer = str(workitem.get('consumer') or '')
            pod_id = socket.gethostname()
            if current_consumer and (CONSUMER_FILTER in current_consumer) and (pod_id in current_consumer):
                print(f"[INFO] Skipping reserved workitem {workitem.get('id')} (consumer={current_consumer})")
                return

        # ?뚰겕?꾩씠??泥섎━ ?쒖옉 濡쒓렇
        try:
            upsert_workitem({
                "id": workitem['id'],
                "log": f"'{workitem['activity_name']}' 업무를 실행합니다."
            }, workitem['tenant_id'])
        except Exception as log_error:
            print(f"[WARNING] Failed to update workitem log: {log_error}")
        
        if workitem['status'] == "SUBMITTED":
            print(f"[DEBUG] Starting safe_handle_workitem for workitem: {workitem['id']}")
            version_tag = workitem.get('version_tag')
            version = workitem.get('version')
            tenant_id = workitem['tenant_id']
            arcv_id = None
            if not version_tag and not version and workitem.get('proc_inst_id'):
                try:
                    process_instance = fetch_process_instance(workitem['proc_inst_id'], tenant_id)
                    if process_instance and getattr(process_instance, "proc_def_version", None):
                        arcv_id = process_instance.proc_def_version
                except Exception as e:
                    print(f"[WARN] Failed to fetch process instance for version: {e}")
            process_definition = fetch_process_definition_by_version(
                workitem['proc_def_id'],
                version_tag,
                version,
                tenant_id,
                arcv_id,
            )
            activities = process_definition.get('activities', [])
            
            task_type = 'userTask'
            for activity in activities:
                if activity.get('id') == workitem['activity_id']:
                    task_type = activity.get('type')
                    break
            
            normalized_task_type = str(task_type or '').lower()
            if normalized_task_type in ('usertask', 'scripttask', 'manualtask', 'callactivity'):
                await handle_workitem(workitem)
            elif normalized_task_type == 'servicetask':
                await handle_service_workitem(workitem)
        elif workitem['status'] == "PENDING":
            print(f"[DEBUG] Starting safe_handle_workitem for pending workitem: {workitem['id']}")
            await handle_pending_workitem(workitem)
        else:
            print(f"[WARNING] Unknown workitem status: {workitem['status']} for workitem: {workitem['id']}")

    except Exception as e:
        print(f"[ERROR] Error in safe_handle_workitem for workitem {workitem['id']}: {str(e)}")
        try:
            workitem['retry'] = workitem.get('retry', 0) + 1
            workitem['consumer'] = None
            if workitem['retry'] >= 3:
                workitem['status'] = "DONE"
                workitem['log'] = f"[Error] Error in safe_handle_workitem for workitem {workitem['id']}: {str(e)}"
                process_instance = fetch_process_instance(workitem['proc_inst_id'], workitem['tenant_id'])
                if process_instance and process_instance.status == "NEW":
                    process_instance.status = "RUNNING"
                    upsert_process_instance(process_instance, workitem['tenant_id'])
                    print(f"[INFO] Updated instance {workitem['proc_inst_id']} status to RUNNING due to workitem failure")
            else:
                workitem['log'] = f"실행하는 중 오류가 발생했습니다. 다시 시도하겠습니다. (시도 {workitem['retry']}/3)"
            upsert_workitem(workitem, workitem['tenant_id'])
        except Exception as update_error:
            print(f"[ERROR] Failed to update workitem error status: {update_error}")
    finally:
        # ?뚰겕?꾩씠??泥섎━ ?꾨즺 ??consumer ?댁젣
        try:
            upsert_workitem({
                "id": workitem['id'],
                "consumer": None
            }, workitem['tenant_id'])
            print(f"[INFO] Released consumer lock for workitem: {workitem['id']}")
        except Exception as e:
            print(f"[ERROR] Failed to release consumer lock for workitem {workitem['id']}: {str(e)}")
        
        # ?쒖뒪???꾨즺 ??異붿쟻 紐⑸줉?먯꽌 ?쒓굅
        if asyncio.current_task() in running_tasks:
            running_tasks.remove(asyncio.current_task())

async def polling_workitem():
    try:
        all_workitems = []
        
        # SUBMITTED ?곹깭 ?뚰겕?꾩씠??議고쉶
        try:
            submitted_workitems = fetch_workitem_with_submitted_status()
            if submitted_workitems:
                all_workitems.extend(submitted_workitems)
                print(f"[DEBUG] Found {len(submitted_workitems)} submitted workitems")
        except Exception as e:
            print(f"[ERROR] Failed to fetch submitted workitems: {str(e)}")
        # PENDING ?곹깭 ?뚰겕?꾩씠??議고쉶
        try:
            pending_workitems = fetch_workitem_with_pending_status()
            if pending_workitems:
                all_workitems.extend(pending_workitems)
                print(f"[DEBUG] Found {len(pending_workitems)} pending workitems")
        except Exception as e:
            print(f"[ERROR] Failed to fetch pending workitems: {str(e)}")

        if len(all_workitems) == 0:
            return

        print(f"[INFO] Processing {len(all_workitems)} workitems")
        tasks = []
        for workitem in all_workitems:
            # consumer ?쒖쇅 洹쒖튃: consumer媛 "CONSUMER_FILTER? pod_id瑜?紐⑤몢 ?ы븿"?섎㈃ ?ㅽ궢
            if CONSUMER_FILTER:
                wi_consumer = str(workitem.get('consumer') or '')
                pod_id = socket.gethostname()
                if wi_consumer and (CONSUMER_FILTER in wi_consumer) and (pod_id in wi_consumer):
                    print(f"[DEBUG] Skip reserved workitem {workitem.get('id')} (consumer={wi_consumer})")
                    continue

            # shutdown ?대깽?멸? ?ㅼ젙?섏뿀?쇰㈃ ???쒖뒪?щ? ?쒖옉?섏? ?딆쓬
            if shutdown_event.is_set():
                print("[INFO] Shutdown in progress, skipping new workitems")
                break
                
            task = asyncio.create_task(safe_handle_workitem(workitem))
            running_tasks.add(task)
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # 寃곌낵 ?뺤씤 諛?濡쒓퉭
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"[ERROR] Task {i} failed: {result}")
                else:
                    print(f"[DEBUG] Task {i} completed successfully")
                    
            
    except Exception as e:
        print(f"[ERROR] Polling workitem failed: {str(e)}")
        # Supabase ?곌껐 ?ㅻ쪟??寃쎌슦 ?좎떆 ?湲?
        if "Supabase client is not configured" in str(e) or "DB fetch failed" in str(e) or "network" in str(e).lower():
            print("[INFO] Database connection error, waiting before retry...")
            await asyncio.sleep(10)
        else:
            # ?ㅻⅨ ?ㅻ쪟??寃쎌슦 吏㏃? ?湲????ъ떆??
            print("[INFO] Other error occurred, waiting before retry...")
            await asyncio.sleep(5)

async def cleanup_task():
    # Periodically clear stale consumers.
    while not shutdown_event.is_set():
        try:
            cleanup_stale_consumers()
            print("[DEBUG] Cleanup task completed successfully")
        except Exception as e:
            print(f"[ERROR] Cleanup task error: {e}")
            # ?ㅻ쪟 諛쒖깮 ??吏㏃? ?湲????ъ떆??
            await asyncio.sleep(60)
            continue
        
        # ?뺤긽?곸씤 寃쎌슦 5遺??湲?
        await asyncio.sleep(300)

async def start_polling():
    try:
        setting_database()
        print("[INFO] Database configuration completed")
    except Exception as e:
        print(f"[ERROR] Failed to configure database: {e}")
        return

    # cleanup ?쒖뒪???쒖옉
    cleanup_task_obj = asyncio.create_task(cleanup_task())
    print("[INFO] Cleanup task started")
    
    # ?뚯씪 ?뺣━ ?대쭅 ?쒖뒪???쒖옉
    file_cleanup_task_obj = asyncio.create_task(file_cleanup_polling_task(shutdown_event, polling_interval=300))
    print("[INFO] File cleanup polling task started")

    while not shutdown_event.is_set():
        try:
            await polling_workitem()
        except Exception as e:
            print(f"[Polling Loop Error] {e}")
            # ?ㅻ쪟 諛쒖깮 ??吏㏃? ?湲?
            await asyncio.sleep(5)
            continue
        
        if shutdown_event.is_set():
            break
            
        await asyncio.sleep(5)
    
    # cleanup ?쒖뒪??痍⑥냼
    print("[INFO] Cancelling cleanup task...")
    cleanup_task_obj.cancel()
    try:
        await cleanup_task_obj
    except asyncio.CancelledError:
        print("[INFO] Cleanup task cancelled successfully")
    except Exception as e:
        print(f"[ERROR] Error cancelling cleanup task: {e}")
    
    # ?뚯씪 ?뺣━ ?쒖뒪??痍⑥냼
    print("[INFO] Cancelling file cleanup task...")
    file_cleanup_task_obj.cancel()
    try:
        await file_cleanup_task_obj
    except asyncio.CancelledError:
        print("[INFO] File cleanup task cancelled successfully")
    except Exception as e:
        print(f"[ERROR] Error cancelling file cleanup task: {e}")

async def graceful_shutdown():
    # Graceful shutdown handler.
    print("[INFO] Starting graceful shutdown...")
    shutdown_event.set()
    
    # 吏꾪뻾 以묒씤 紐⑤뱺 ?쒖뒪?ш? ?꾨즺???뚭퉴吏 ?湲?
    if running_tasks:
        print(f"[INFO] Waiting for {len(running_tasks)} running tasks to complete...")
        await asyncio.gather(*running_tasks, return_exceptions=True)
        print("[INFO] All running tasks completed")
    
    print("[INFO] Graceful shutdown completed")

def signal_handler(signum, frame):
    # Signal handler.
    print(f"[INFO] Received signal {signum}, initiating graceful shutdown...")
    asyncio.create_task(graceful_shutdown())

def run_polling_service():
    try:
        # ?쒓렇???몃뱾???깅줉
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        print("[INFO] Starting polling service with graceful shutdown support...")
        asyncio.run(start_polling())
    except KeyboardInterrupt:
        print("[INFO] Polling service stopped by user")
    except Exception as e:
        print(f"[ERROR] Polling service failed: {str(e)}")
        raise e

if __name__ == "__main__":
    run_polling_service() 
