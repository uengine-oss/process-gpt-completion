from database import insert_usage
from fastapi import HTTPException

# 사용량 기록
def usage(raw_data):
    try:
        # raw_data: {
        # "serviceId":       "CHAT_LLM", // service.id
        # "tenantId":        "localhost", // tenant.id
        # "userId":          "gpt@gpt.org", // user
        # "startAt":         "2025-08-06T09:00:00+09:00", // service start at.
        # "usage": {
        #     "<model_alias>": { "request": 100, "response": 200, "cachedRequest": 100 }
        # },
        # "process_def_id":  null,
        # "process_inst_id": null,
        # "agent_id":        null
        # }
        return insert_usage(raw_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
