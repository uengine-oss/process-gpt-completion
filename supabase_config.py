"""Supabase 접속 설정을 환경변수에서 읽는다.

이 파일은 저장소에 없던 것을 로컬 실행을 위해 복구한 것이다. `database.py` 와
`process_db_manager.py` 가 import 하는데 git 에 추적된 적이 없어(untracked) 어느 시점에
사라졌고, 그 뒤로는 이미 로드된 프로세스만 살아 있었다.

같은 저장소의 다른 서비스(`polling_service/database.py`, `fcm_service/database.py`)가
`os.getenv("SUPABASE_URL")` / `os.getenv("SUPABASE_KEY")` 를 그대로 읽으므로 여기서도
같은 이름을 쓴다.
"""

from __future__ import annotations

import os

# 로컬 supabase 스택의 기본 JWT 시크릿. 운영에서는 반드시 환경변수로 주입된다.
_LOCAL_JWT_SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"


def get_supabase_url() -> str | None:
    return os.getenv("SUPABASE_URL")


def get_supabase_key() -> str | None:
    return os.getenv("SUPABASE_KEY")


def get_supabase_jwt_secret() -> str:
    """로컬 발급 토큰을 검증할 때 쓰는 시크릿.

    운영에서 값이 비면 로컬 기본값으로 토큰을 검증하게 되어 위험하므로, ENV=production
    에서는 환경변수가 없으면 그대로 비워 돌려준다(호출부가 검증에 실패한다).
    """
    secret = os.getenv("SUPABASE_JWT_SECRET") or os.getenv("JWT_SECRET")
    if secret:
        return secret
    if (os.getenv("ENV") or "").strip().lower() == "production":
        return ""
    return _LOCAL_JWT_SECRET
