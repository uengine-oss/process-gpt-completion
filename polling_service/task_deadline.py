from datetime import datetime, time
from typing import Any

import pytz


KST = pytz.timezone("Asia/Seoul")


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return KST.localize(parsed)
    return parsed.astimezone(KST)


def ensure_minimum_task_due_date(workitem_data: dict, instance_start: Any = None) -> dict:
    """Keep a new task from becoming overdue on its process creation day."""
    if not isinstance(workitem_data, dict):
        return workitem_data
    task_start = _as_datetime(workitem_data.get("start_date"))
    process_start = _as_datetime(instance_start)
    start = max((d for d in (task_start, process_start) if d is not None), default=None)
    if start is None:
        return workitem_data
    minimum = KST.localize(datetime.combine(start.date(), time.max))
    due = _as_datetime(workitem_data.get("due_date"))
    if due is None or due < minimum:
        workitem_data["due_date"] = minimum.isoformat()
    return workitem_data
