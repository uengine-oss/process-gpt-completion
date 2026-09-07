import asyncio
import os
import time
import uuid
from typing import Any, Dict, Optional
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

import jwt
import requests
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from database import (
    approve_admin_request,
    check_tenant_owner,
    create_admin_request,
    create_user,
    fetch_user_info_by_uid_and_tenant,
    get_admin_requests,
    get_my_admin_requests,
    invite_user,
    reject_admin_request,
    set_initial_info,
    subdomain_var,
    supabase_client_var,
    update_user_admin,
)
from supabase_config import get_supabase_jwt_secret, get_supabase_key, get_supabase_url

AUTH_TIMEOUT_SECONDS = 5
DEFAULT_AUTH_TOKEN_EXPIRY_SECONDS = 3600
SSO_EXCHANGE_STATUS_AUTHENTICATED = "authenticated"
SSO_EXCHANGE_STATUS_SUPABASE_NOT_CONFIGURED = "supabase_not_configured"
SSO_HEADER_EMPLOYEE_NO = "x-playground-employeeno"
SSO_HEADER_EMAIL = "x-playground-email"
SSO_HEADER_USERNAME = "x-playground-username"
SSO_HEADER_ORG_CODE = "x-playground-org-code"
SSO_HEADER_ORG_NAME = "x-playground-org-name"
LOCAL_SSO_DEFAULTS = {
    "employee_no": "10001234",
    "email": "user@example.com",
    "user_name": "홍길동",
    "org_code": "D001",
    "org_name": "Platform Team",
}


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


def _decode_local_supabase_token(access_token: str) -> Optional[Dict[str, Any]]:
    jwt_secret = get_supabase_jwt_secret()

    try:
        payload = jwt.decode(
            access_token,
            jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except jwt.PyJWTError:
        return None

    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id.strip():
        return None

    app_metadata = payload.get("app_metadata")
    user_metadata = payload.get("user_metadata")

    return {
        "id": user_id.strip(),
        "email": payload.get("email"),
        "app_metadata": app_metadata if isinstance(app_metadata, dict) else {},
        "user_metadata": user_metadata if isinstance(user_metadata, dict) else {},
        "role": payload.get("role"),
    }


def _fetch_authenticated_user(access_token: str) -> Dict[str, Any]:
    locally_decoded = _decode_local_supabase_token(access_token)
    if locally_decoded is not None:
        return locally_decoded

    supabase_url = get_supabase_url()
    supabase_key = get_supabase_key()

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


def _get_supabase_client():
    supabase = supabase_client_var.get()
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase client is not configured for this request")
    return supabase


def _has_supabase_client() -> bool:
    return supabase_client_var.get() is not None


def _get_service_role_headers() -> Dict[str, str]:
    supabase_key = get_supabase_key()
    if not supabase_key:
        raise HTTPException(status_code=500, detail="Supabase service role key is not configured")
    return {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
    }


def _get_auth_token_expiry_seconds() -> int:
    raw_expiry = os.getenv("SUPABASE_JWT_EXPIRY", str(DEFAULT_AUTH_TOKEN_EXPIRY_SECONDS))
    try:
        expiry = int(raw_expiry)
    except ValueError:
        expiry = DEFAULT_AUTH_TOKEN_EXPIRY_SECONDS
    return max(expiry, 300)


def _validate_redirect_target(redirect_to: Optional[str]) -> Optional[str]:
    if redirect_to is None:
        return None
    target = redirect_to.strip()
    if not target:
        return None
    if not target.startswith("/") or target.startswith("//"):
        raise HTTPException(status_code=400, detail="redirect_to must be a relative path")
    return target


def _append_sso_status_to_redirect_target(redirect_to: str, status: str) -> str:
    parsed = urlsplit(redirect_to)
    query_items = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key != "sso_status"]
    query_items.append(("sso_status", status))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query_items), parsed.fragment))


def _is_local_development() -> bool:
    return os.getenv("ENV") != "production"


def _get_local_sso_identity_defaults() -> Dict[str, str]:
    return {
        "employee_no": (os.getenv("LOCAL_SSO_EMPLOYEE_NO") or LOCAL_SSO_DEFAULTS["employee_no"]).strip(),
        "email": (os.getenv("LOCAL_SSO_EMAIL") or LOCAL_SSO_DEFAULTS["email"]).strip().lower(),
        "user_name": (os.getenv("LOCAL_SSO_USERNAME") or LOCAL_SSO_DEFAULTS["user_name"]).strip(),
        "org_code": (os.getenv("LOCAL_SSO_ORG_CODE") or LOCAL_SSO_DEFAULTS["org_code"]).strip(),
        "org_name": (os.getenv("LOCAL_SSO_ORG_NAME") or LOCAL_SSO_DEFAULTS["org_name"]).strip(),
    }


def _extract_sso_identity(request: Request) -> Dict[str, str]:
    employee_no = (request.headers.get(SSO_HEADER_EMPLOYEE_NO) or "").strip()
    email = (request.headers.get(SSO_HEADER_EMAIL) or "").strip().lower()
    user_name = unquote((request.headers.get(SSO_HEADER_USERNAME) or "").strip())
    org_code = (request.headers.get(SSO_HEADER_ORG_CODE) or "").strip()
    org_name = unquote((request.headers.get(SSO_HEADER_ORG_NAME) or "").strip())

    if _is_local_development():
        local_defaults = _get_local_sso_identity_defaults()
        employee_no = employee_no or local_defaults["employee_no"]
        email = email or local_defaults["email"]
        user_name = user_name or local_defaults["user_name"]
        org_code = org_code or local_defaults["org_code"]
        org_name = org_name or local_defaults["org_name"]

    if not employee_no:
        raise HTTPException(status_code=401, detail=f"Missing required SSO header: {SSO_HEADER_EMPLOYEE_NO}")
    if not email:
        raise HTTPException(status_code=401, detail=f"Missing required SSO header: {SSO_HEADER_EMAIL}")

    if not user_name:
        user_name = email.split("@")[0]

    return {
        "employee_no": employee_no,
        "email": email,
        "user_name": user_name,
        "org_code": org_code,
        "org_name": org_name,
    }


def _find_auth_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    supabase_url = get_supabase_url()
    if not supabase_url:
        raise HTTPException(status_code=500, detail="Supabase URL is not configured")

    headers = _get_service_role_headers()
    page = 1
    per_page = 200

    while page <= 10:
        try:
            response = requests.get(
                f"{supabase_url.rstrip('/')}/auth/v1/admin/users",
                headers=headers,
                params={"page": page, "per_page": per_page},
                timeout=AUTH_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail="Failed to query Supabase admin users") from exc

        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail="Failed to query Supabase admin users")

        try:
            payload = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="Invalid Supabase admin user response") from exc

        users = payload.get("users") if isinstance(payload, dict) else None
        if not isinstance(users, list):
            return None

        for user in users:
            if isinstance(user, dict) and (user.get("email") or "").strip().lower() == email:
                return user

        if len(users) < per_page:
            break
        page += 1

    return None


def _find_public_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    supabase = _get_supabase_client()
    response = supabase.table("users").select("*").eq("email", email).limit(1).execute()
    if response.data:
        return response.data[0]
    return None


def _ensure_sso_user(identity: Dict[str, str], tenant_id: str) -> Dict[str, Any]:
    supabase = _get_supabase_client()
    existing_user = _find_public_user_by_email(identity["email"])
    user_id = existing_user.get("id") if existing_user else None

    if not user_id:
        try:
            response = supabase.auth.admin.create_user(
                {
                    "email": identity["email"],
                    "password": uuid.uuid4().hex,
                    "email_confirm": True,
                    "user_metadata": {
                        "name": identity["user_name"],
                    },
                    "app_metadata": {
                        "tenant_id": tenant_id,
                        "employee_no": identity["employee_no"],
                        "org_code": identity["org_code"],
                        "org_name": identity["org_name"],
                        "provider": "playground-sso",
                    },
                }
            )
            created_user = getattr(response, "user", None)
            user_id = getattr(created_user, "id", None) if created_user is not None else None
        except Exception:
            matched_user = _find_auth_user_by_email(identity["email"])
            user_id = matched_user.get("id") if matched_user else None

    if not user_id:
        raise HTTPException(status_code=500, detail="Failed to provision Supabase user for SSO exchange")

    membership = _fetch_user_membership(user_id, tenant_id)
    role = membership.get("role") if membership else "user"
    is_admin = _normalize_bool(membership.get("is_admin")) if membership else False

    if _is_tenant_owner(user_id, tenant_id):
        is_admin = True
        if not role:
            role = "superadmin"

    membership_payload = {
        "id": user_id,
        "email": identity["email"],
        "username": identity["user_name"],
        "role": role or "user",
        "is_admin": is_admin,
        "employee_no": identity["employee_no"],
        "org_code": identity["org_code"],
        "org_name": identity["org_name"],
        "tenant_id": tenant_id,
    }

    if membership:
        (
            supabase.table("users")
            .update(
                {
                    "email": membership_payload["email"],
                    "username": membership_payload["username"],
                    "role": membership_payload["role"],
                    "is_admin": membership_payload["is_admin"],
                    "employee_no": membership_payload["employee_no"],
                    "org_code": membership_payload["org_code"],
                    "org_name": membership_payload["org_name"],
                }
            )
            .eq("id", user_id)
            .eq("tenant_id", tenant_id)
            .execute()
        )
    else:
        supabase.table("users").insert(membership_payload).execute()

    try:
        supabase.auth.admin.update_user_by_id(
            user_id,
            {
                "user_metadata": {
                    "name": identity["user_name"],
                },
                "app_metadata": {
                    "tenant_id": tenant_id,
                    "employee_no": identity["employee_no"],
                    "org_code": identity["org_code"],
                    "org_name": identity["org_name"],
                    "provider": "playground-sso",
                    "membership_role": membership_payload["role"],
                    "is_admin": membership_payload["is_admin"],
                },
            },
        )
    except Exception as exc:
        print(f"[auth-exchange] Failed to sync auth metadata for {user_id}: {exc}")

    return membership_payload


def _issue_sso_exchange_token(user_info: Dict[str, Any], identity: Dict[str, str], tenant_id: str) -> Dict[str, Any]:
    jwt_secret = get_supabase_jwt_secret()

    now = int(time.time())
    expires_in = _get_auth_token_expiry_seconds()
    exp = now + expires_in
    session_id = str(uuid.uuid4())

    user_role = user_info.get("role") or ("admin" if _normalize_bool(user_info.get("is_admin")) else "viewer")

    claims = {
        "iss": "process-gpt-backend",
        "aud": "authenticated",
        "sub": user_info["id"],
        "email": user_info["email"],
        "role": "authenticated",
        "user_role": user_role,
        "aal": "aal1",
        "amr": [{"method": "sso-header", "timestamp": now}],
        "session_id": session_id,
        "is_anonymous": False,
        "iat": now,
        "exp": exp,
        "app_metadata": {
            "provider": "playground-sso",
            "providers": ["playground-sso"],
            "tenant_id": tenant_id,
            "employee_no": identity["employee_no"],
            "org_code": identity["org_code"],
            "org_name": identity["org_name"],
            "membership_role": user_role,
            "is_admin": _normalize_bool(user_info.get("is_admin")),
        },
        "user_metadata": {
            "name": identity["user_name"],
        },
    }

    access_token = jwt.encode(claims, jwt_secret, algorithm="HS256")
    return {
        "access_token": access_token,
        "expires_in": expires_in,
        "expires_at": exp,
        "session_id": session_id,
    }


def _build_sso_user_payload(identity: Dict[str, str], user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "id": None,
        "email": identity["email"],
        "username": identity["user_name"],
        "role": None,
        "is_admin": False,
        "employee_no": identity["employee_no"],
        "org_code": identity["org_code"],
        "org_name": identity["org_name"],
    }

    if user_info:
        payload["id"] = user_info.get("id")
        payload["email"] = user_info.get("email") or payload["email"]
        payload["username"] = user_info.get("username") or payload["username"]
        payload["role"] = user_info.get("role") or "user"
        payload["is_admin"] = _normalize_bool(user_info.get("is_admin"))
        payload["employee_no"] = user_info.get("employee_no") or payload["employee_no"]
        payload["org_code"] = user_info.get("org_code") or payload["org_code"]
        payload["org_name"] = user_info.get("org_name") or payload["org_name"]

    return payload


def _build_sso_exchange_payload(
    status: str,
    tenant_id: str,
    identity: Dict[str, str],
    user_info: Optional[Dict[str, Any]] = None,
    token_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "status": status,
        "tenant_id": tenant_id,
        "user": _build_sso_user_payload(identity, user_info),
    }

    if token_info:
        payload.update(
            {
                "access_token": token_info["access_token"],
                "token_type": "bearer",
                "expires_in": token_info["expires_in"],
                "expires_at": token_info["expires_at"],
            }
        )

    return payload


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


def _is_tenant_owner(user_id: str, tenant_id: str) -> bool:
    try:
        return check_tenant_owner(tenant_id, user_id)
    except HTTPException:
        return False


def _is_tenant_admin(user_id: str, tenant_id: str) -> bool:
    if not tenant_id:
        return False

    if _is_tenant_owner(user_id, tenant_id):
        return True

    membership = _fetch_user_membership(user_id, tenant_id)
    if not membership:
        return False

    role = membership.get("role")
    normalized_role = role.strip().lower() if isinstance(role, str) else ""
    return _normalize_bool(membership.get("is_admin")) or normalized_role in {"admin", "superadmin"}


def _is_super_admin(user_id: str, tenant_id: str) -> bool:
    if not tenant_id:
        return False

    if _is_tenant_owner(user_id, tenant_id):
        return True

    membership = _fetch_user_membership(user_id, tenant_id)
    if not membership:
        return False

    role = membership.get("role")
    return isinstance(role, str) and role.lower() == "superadmin"


def _can_access_tenant(user_id: str, tenant_id: str) -> bool:
    if not tenant_id:
        return False

    if _is_tenant_owner(user_id, tenant_id):
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


async def exchange_sso_token(request: Request):
    tenant_id = subdomain_var.get()
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise HTTPException(status_code=400, detail="tenant_id is not resolved for this request")

    tenant_id = tenant_id.strip()
    identity = _extract_sso_identity(request)
    redirect_to = _validate_redirect_target(request.query_params.get("redirect_to"))

    if not _has_supabase_client():
        if redirect_to:
            response = RedirectResponse(
                url=_append_sso_status_to_redirect_target(
                    redirect_to,
                    SSO_EXCHANGE_STATUS_SUPABASE_NOT_CONFIGURED,
                ),
                status_code=303,
            )
        else:
            response = JSONResponse(
                content=_build_sso_exchange_payload(
                    status=SSO_EXCHANGE_STATUS_SUPABASE_NOT_CONFIGURED,
                    tenant_id=tenant_id,
                    identity=identity,
                )
            )

        response.delete_cookie(key="access_token", path="/")
        return response

    user_info = await asyncio.to_thread(_ensure_sso_user, identity, tenant_id)
    token_info = _issue_sso_exchange_token(user_info, identity, tenant_id)

    if redirect_to:
        response = RedirectResponse(url=redirect_to, status_code=303)
    else:
        response = JSONResponse(
            content=_build_sso_exchange_payload(
                status=SSO_EXCHANGE_STATUS_AUTHENTICATED,
                tenant_id=tenant_id,
                identity=identity,
                user_info=user_info,
                token_info=token_info,
            )
        )

    secure_cookie = os.getenv("ENV") == "production"
    response.set_cookie(
        key="access_token",
        value=token_info["access_token"],
        max_age=token_info["expires_in"],
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        path="/",
    )
    return response

# ============================================
# Admin Request Handlers (권한 신청/승인)
# ============================================

async def handle_create_admin_request(request: Request):
    json_data = await request.json()
    authenticated_user = await _require_authenticated_user(request)
    input_data = _extract_input(json_data)
    tenant_id = subdomain_var.get()

    user_id = authenticated_user["id"]
    email = authenticated_user.get("email") or ""
    username = input_data.get("username") or authenticated_user.get("user_metadata", {}).get("name") or ""
    reason = input_data.get("reason") or ""
    requested_role = input_data.get("requested_role") or "admin"

    if not isinstance(reason, str):
        raise HTTPException(status_code=400, detail="reason must be a string")

    return create_admin_request(
        user_id=user_id,
        email=email,
        username=username.strip() if username else "",
        tenant_id=tenant_id,
        reason=reason.strip(),
        requested_role=requested_role,
    )


async def handle_list_admin_requests(request: Request):
    json_data = await request.json()
    authenticated_user = await _require_authenticated_user(request)
    input_data = _extract_input(json_data)
    tenant_id = subdomain_var.get()

    if not _is_tenant_admin(authenticated_user["id"], tenant_id):
        raise HTTPException(status_code=403, detail="Admin 권한이 필요합니다.")

    status = input_data.get("status") or "all"
    return get_admin_requests(tenant_id, status)


async def handle_my_admin_requests(request: Request):
    authenticated_user = await _require_authenticated_user(request)
    tenant_id = subdomain_var.get()
    return get_my_admin_requests(authenticated_user["id"], tenant_id)


async def handle_approve_admin_request(request: Request):
    json_data = await request.json()
    authenticated_user = await _require_authenticated_user(request)
    input_data = _extract_input(json_data)
    tenant_id = subdomain_var.get()

    if not _is_tenant_admin(authenticated_user["id"], tenant_id):
        raise HTTPException(status_code=403, detail="Admin 권한이 필요합니다.")

    request_id = input_data.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise HTTPException(status_code=400, detail="request_id is required")

    reviewer = authenticated_user.get("email") or "admin"
    return approve_admin_request(request_id.strip(), reviewer, tenant_id)


async def handle_reject_admin_request(request: Request):
    json_data = await request.json()
    authenticated_user = await _require_authenticated_user(request)
    input_data = _extract_input(json_data)
    tenant_id = subdomain_var.get()

    if not _is_tenant_admin(authenticated_user["id"], tenant_id):
        raise HTTPException(status_code=403, detail="Admin 권한이 필요합니다.")

    request_id = input_data.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise HTTPException(status_code=400, detail="request_id is required")

    reject_reason = input_data.get("reject_reason") or ""
    reviewer = authenticated_user.get("email") or "admin"
    return reject_admin_request(request_id.strip(), reviewer, tenant_id, reject_reason)


async def handle_refresh_role_token(request: Request):
    """현재 사용자의 DB 최신 role로 JWT를 재발행한다."""
    authenticated_user = await _require_authenticated_user(request)
    tenant_id = subdomain_var.get()

    if not _has_supabase_client():
        raise HTTPException(status_code=500, detail="Supabase client is not configured")

    supabase = supabase_client_var.get()
    user_id = authenticated_user["id"]

    # DB에서 최신 role 조회
    user_row = (
        supabase.table("users")
        .select("id, email, username, role, is_admin, profile")
        .eq("id", user_id)
        .eq("tenant_id", tenant_id)
        .limit(1)
        .execute()
    )
    if not user_row.data:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    user_info = user_row.data[0]

    # SSO identity 재구성
    identity = {
        "user_name": user_info.get("username") or "",
        "email": user_info.get("email") or "",
        "employee_no": authenticated_user.get("app_metadata", {}).get("employee_no", ""),
        "org_code": authenticated_user.get("app_metadata", {}).get("org_code", ""),
        "org_name": authenticated_user.get("app_metadata", {}).get("org_name", ""),
    }

    token_info = _issue_sso_exchange_token(user_info, identity, tenant_id)

    return JSONResponse(content={
        "access_token": token_info["access_token"],
        "expires_in": token_info["expires_in"],
        "expires_at": token_info["expires_at"],
        "user": {
            "id": user_info["id"],
            "email": user_info.get("email", ""),
            "username": user_info.get("username", ""),
            "role": user_info.get("role") or "viewer",
            "is_admin": user_info.get("is_admin", False),
        },
    })


def add_routes_to_app(app) :
    app.add_api_route("/set-tenant", combine_input_with_tenant_id, methods=["POST"])
    app.add_api_route("/create-user", combine_input_with_new_user_info, methods=["POST"])
    app.add_api_route("/invite-user", combine_input_with_invite_user_info, methods=["POST"])
    app.add_api_route("/set-initial-info", combine_input_with_set_initial_info, methods=["POST"])
    app.add_api_route("/update-user", combine_input_with_user_info, methods=["POST"])
    app.add_api_route("/auth/sso/exchange", exchange_sso_token, methods=["GET", "POST"])
    app.add_api_route("/auth/sso/refresh-role", handle_refresh_role_token, methods=["POST"])
    app.add_api_route("/admin-requests/create", handle_create_admin_request, methods=["POST"])
    app.add_api_route("/admin-requests/list", handle_list_admin_requests, methods=["POST"])
    app.add_api_route("/admin-requests/my", handle_my_admin_requests, methods=["GET", "POST"])
    app.add_api_route("/admin-requests/approve", handle_approve_admin_request, methods=["POST"])
    app.add_api_route("/admin-requests/reject", handle_reject_admin_request, methods=["POST"])


"""
"""
