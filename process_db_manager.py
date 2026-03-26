import asyncio
import os
from typing import Any, Dict, Optional

import requests
from fastapi import HTTPException, Request

from database import (
    check_tenant_owner,
    create_user,
    fetch_user_info_by_uid_and_tenant,
    invite_user,
    set_initial_info,
    subdomain_var,
    update_user_admin,
)

AUTH_TIMEOUT_SECONDS = 5


def _extract_input(json_data: Any) -> Dict[str, Any]:
    if not isinstance(json_data, dict):
        raise HTTPException(status_code=400, detail="Invalid request body")

    input_data = json_data.get("input")
    if not isinstance(input_data, dict):
        raise HTTPException(status_code=400, detail="Request input is required")

    return input_data


def _extract_access_token(request: Request) -> Optional[str]:
    authorization_header = request.headers.get("authorization")
    if authorization_header:
        scheme, _, token = authorization_header.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()

    cookie_token = request.cookies.get("access_token")
    if cookie_token:
        return cookie_token.strip()

    return None


def _fetch_authenticated_user(access_token: str) -> Dict[str, Any]:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Supabase authentication is not configured")

    try:
        response = requests.get(
            f"{supabase_url.rstrip('/')}/auth/v1/user",
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {access_token}",
            },
            timeout=AUTH_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="Failed to verify access token") from exc

    if response.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="Authentication required")
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail="Failed to verify access token")

    try:
        user = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Invalid authentication response") from exc

    user_id = user.get("id") if isinstance(user, dict) else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    return user


async def _require_authenticated_user(request: Request) -> Dict[str, Any]:
    access_token = _extract_access_token(request)
    if not access_token:
        raise HTTPException(status_code=401, detail="Authentication required")

    return await asyncio.to_thread(_fetch_authenticated_user, access_token)


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _require_same_user(target_user_id: Any, authenticated_user_id: str) -> None:
    if not isinstance(target_user_id, str) or target_user_id != authenticated_user_id:
        raise HTTPException(status_code=403, detail="You can only update your own account")


def _fetch_user_membership(user_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
    try:
        return fetch_user_info_by_uid_and_tenant(user_id, tenant_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            return None
        raise


def _is_tenant_admin(user_id: str, tenant_id: str) -> bool:
    if not tenant_id:
        return False

    if check_tenant_owner(tenant_id, user_id):
        return True

    membership = _fetch_user_membership(user_id, tenant_id)
    if not membership:
        return False

    role = membership.get("role")
    return _normalize_bool(membership.get("is_admin")) or (
        isinstance(role, str) and role.lower() == "superadmin"
    )


def _can_access_tenant(user_id: str, tenant_id: str) -> bool:
    if not tenant_id:
        return False

    if check_tenant_owner(tenant_id, user_id):
        return True

    return _fetch_user_membership(user_id, tenant_id) is not None


def _sanitize_set_tenant_input(input_data: Dict[str, Any], authenticated_user_id: str) -> Dict[str, Any]:
    _require_same_user(input_data.get("user_id"), authenticated_user_id)

    user_info = input_data.get("user_info")
    app_metadata = user_info.get("app_metadata") if isinstance(user_info, dict) else None
    tenant_id = app_metadata.get("tenant_id") if isinstance(app_metadata, dict) else None

    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise HTTPException(status_code=400, detail="tenant_id is required")

    tenant_id = tenant_id.strip()
    if not _can_access_tenant(authenticated_user_id, tenant_id):
        raise HTTPException(status_code=403, detail="You do not have access to this tenant")

    return {
        "user_id": authenticated_user_id,
        "user_info": {
            "app_metadata": {
                "tenant_id": tenant_id
            }
        }
    }


def _sanitize_create_user_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    username = input_data.get("username")
    email = input_data.get("email")
    role = input_data.get("role") or "user"

    if not isinstance(username, str) or not username.strip():
        raise HTTPException(status_code=400, detail="username is required")
    if not isinstance(email, str) or not email.strip():
        raise HTTPException(status_code=400, detail="email is required")
    if not isinstance(role, str) or not role.strip():
        raise HTTPException(status_code=400, detail="role is required")

    return {
        "username": username.strip(),
        "email": email.strip(),
        "role": role.strip(),
    }


def _sanitize_invite_user_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    email = input_data.get("email")
    tenant_id = input_data.get("tenant_id") or subdomain_var.get()

    if not isinstance(email, str) or not email.strip():
        raise HTTPException(status_code=400, detail="email is required")
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise HTTPException(status_code=400, detail="tenant_id is required")

    return {
        "email": email.strip(),
        "is_admin": _normalize_bool(input_data.get("is_admin")),
        "tenant_id": tenant_id.strip(),
    }


def _sanitize_initial_info_input(input_data: Dict[str, Any], authenticated_user_id: str) -> Dict[str, Any]:
    _require_same_user(input_data.get("user_id"), authenticated_user_id)

    user_name = input_data.get("user_name")
    password = input_data.get("password")

    if not isinstance(user_name, str) or not user_name.strip():
        raise HTTPException(status_code=400, detail="user_name is required")
    if not isinstance(password, str) or not password.strip():
        raise HTTPException(status_code=400, detail="password is required")

    return {
        "user_id": authenticated_user_id,
        "user_name": user_name.strip(),
        "password": password,
    }


def _sanitize_update_user_input(input_data: Dict[str, Any], authenticated_user_id: str) -> Dict[str, Any]:
    _require_same_user(input_data.get("user_id"), authenticated_user_id)

    user_info = input_data.get("user_info")
    if not isinstance(user_info, dict):
        raise HTTPException(status_code=400, detail="user_info is required")

    unsupported_keys = set(user_info.keys()) - {"email", "password", "user_metadata"}
    if unsupported_keys:
        raise HTTPException(status_code=400, detail="Unsupported user update fields")

    sanitized_user_info: Dict[str, Any] = {}

    if "email" in user_info:
        email = user_info.get("email")
        if not isinstance(email, str) or not email.strip():
            raise HTTPException(status_code=400, detail="email must be a non-empty string")
        sanitized_user_info["email"] = email.strip()

    if "password" in user_info:
        password = user_info.get("password")
        if not isinstance(password, str) or not password.strip():
            raise HTTPException(status_code=400, detail="password must be a non-empty string")
        sanitized_user_info["password"] = password

    if "user_metadata" in user_info:
        user_metadata = user_info.get("user_metadata")
        if not isinstance(user_metadata, dict):
            raise HTTPException(status_code=400, detail="user_metadata must be an object")

        unsupported_metadata_keys = set(user_metadata.keys()) - {"name"}
        if unsupported_metadata_keys:
            raise HTTPException(status_code=400, detail="Unsupported user_metadata fields")

        if "name" in user_metadata:
            name = user_metadata.get("name")
            if not isinstance(name, str) or not name.strip():
                raise HTTPException(status_code=400, detail="name must be a non-empty string")
            sanitized_user_info["user_metadata"] = {"name": name.strip()}

    if not sanitized_user_info:
        raise HTTPException(status_code=400, detail="At least one allowed user field is required")

    return {
        "user_id": authenticated_user_id,
        "user_info": sanitized_user_info,
    }

async def combine_input_with_tenant_id(request: Request):
    json_data = await request.json()
    authenticated_user = await _require_authenticated_user(request)
    input_data = _extract_input(json_data)
    return update_user_admin(_sanitize_set_tenant_input(input_data, authenticated_user["id"]))

async def combine_input_with_new_user_info(request: Request):
    json_data = await request.json()
    authenticated_user = await _require_authenticated_user(request)
    input_data = _extract_input(json_data)
    tenant_id = subdomain_var.get()

    if not _is_tenant_admin(authenticated_user["id"], tenant_id):
        raise HTTPException(status_code=403, detail="Admin privileges are required")

    return create_user(_sanitize_create_user_input(input_data))

async def combine_input_with_invite_user_info(request: Request):
    json_data = await request.json()
    authenticated_user = await _require_authenticated_user(request)
    input_data = _extract_input(json_data)
    sanitized_input = _sanitize_invite_user_input(input_data)

    if not _is_tenant_admin(authenticated_user["id"], sanitized_input["tenant_id"]):
        raise HTTPException(status_code=403, detail="Admin privileges are required")

    return invite_user(sanitized_input)

async def combine_input_with_set_initial_info(request: Request):
    json_data = await request.json()
    authenticated_user = await _require_authenticated_user(request)
    input_data = _extract_input(json_data)
    return set_initial_info(_sanitize_initial_info_input(input_data, authenticated_user["id"]))

async def combine_input_with_user_info(request: Request):
    json_data = await request.json()
    authenticated_user = await _require_authenticated_user(request)
    input_data = _extract_input(json_data)
    return update_user_admin(_sanitize_update_user_input(input_data, authenticated_user["id"]))

def add_routes_to_app(app) :
    app.add_api_route("/set-tenant", combine_input_with_tenant_id, methods=["POST"])
    app.add_api_route("/create-user", combine_input_with_new_user_info, methods=["POST"])
    app.add_api_route("/invite-user", combine_input_with_invite_user_info, methods=["POST"])
    app.add_api_route("/set-initial-info", combine_input_with_set_initial_info, methods=["POST"])
    app.add_api_route("/update-user", combine_input_with_user_info, methods=["POST"])


"""
"""
