"""
A2A 에이전트 카드 가져오기 (`GET /multi-agent/fetch-data`).

조직도에서 A2A 에이전트 주소를 넣고 「가져오기」를 누르면 이 서버가 그 주소의 에이전트 카드를 대신 읽는다.
서버가 사용자가 준 주소로 요청을 보내므로, 예전처럼 주소를 그대로 이어 붙이면
- `http://내부서비스/비밀?x=` 처럼 뒤에 붙는 `/.well-known/agent.json` 을 질의 문자열로 삼켜 내부의 아무 JSON 이나 읽고,
- 클라우드 메타데이터(169.254.169.254)나 이 컨테이너 자신(127.0.0.1)에 닿고,
- 상대가 돌려준 JSON 을 그대로 화면에 넘긴다.
여기서는 주소를 scheme·host·경로로만 다시 만들고, 닿으면 안 되는 주소를 막고, 카드 모양인 응답의 카드 필드만 돌려준다.

사설망(10/8, 172.16/12, 192.168/16) 은 기본으로 허용한다 — 설치본은 A2A 에이전트를 같은 도커 망이나 사내망에 둔다.
여러 테넌트가 함께 쓰는 서버는 `A2A_FETCH_ALLOW_PRIVATE=false` 로 막는다.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlsplit, urlunsplit

import requests

# A2A 0.3 부터 카드 위치가 agent-card.json, 그 전 판은 agent.json
CARD_PATHS = ("/.well-known/agent-card.json", "/.well-known/agent.json")
CARD_FIELDS = (
    "name", "description", "url", "version", "protocolVersion", "provider", "capabilities",
    "defaultInputModes", "defaultOutputModes", "skills", "preferredTransport", "additionalInterfaces",
    "documentationUrl", "iconUrl", "supportsAuthenticatedExtendedCard", "securitySchemes", "security",
)
SKILL_FIELDS = ("id", "name", "description", "tags", "examples", "inputModes", "outputModes")
MAX_BYTES = 512 * 1024
TIMEOUT = (3.05, 10)


class CardFetchError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def allow_private() -> bool:
    return os.getenv("A2A_FETCH_ALLOW_PRIVATE", "true").strip().lower() not in ("0", "false", "no", "off")


def card_urls(agent_url: str) -> List[str]:
    """사용자가 준 주소에서 카드 주소 후보를 만든다. 질의·조각·계정 정보는 받지 않는다."""
    raw = (agent_url or "").strip()
    if not raw:
        raise CardFetchError(400, "agent_url is required")
    if "://" not in raw:
        raw = "http://" + raw
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https"):
        raise CardFetchError(400, "agent_url must be http or https")
    if not parts.hostname:
        raise CardFetchError(400, "agent_url has no host")
    if parts.username or parts.password:
        raise CardFetchError(400, "agent_url must not contain credentials")
    if parts.query or parts.fragment:
        raise CardFetchError(400, "agent_url must not contain a query or fragment")
    try:
        parts.port
    except ValueError:
        raise CardFetchError(400, "agent_url has an invalid port")

    path = parts.path.rstrip("/")
    for card in CARD_PATHS:
        if path.endswith(card):
            # 카드 주소를 그대로 넣은 경우
            return [urlunsplit((parts.scheme, parts.netloc, path, "", ""))]
    bases = [path] + ([""] if path else [])
    return [urlunsplit((parts.scheme, parts.netloc, base + card, "", "")) for base in bases for card in CARD_PATHS]


def _blocked(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address], private_ok: bool) -> Optional[str]:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if ip.is_loopback or ip.is_unspecified:
        return "loopback"
    if ip.is_link_local:
        return "link-local (cloud metadata)"
    if ip.is_multicast or ip.is_reserved:
        return "reserved"
    if ip.is_private and not private_ok:
        return "private network"
    return None


def check_host(host: str, port: int) -> None:
    """이름이 가리키는 주소를 모두 보고, 하나라도 막힌 주소면 거절한다."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise CardFetchError(400, f"cannot resolve host {host}")
    private_ok = allow_private()
    for info in infos:
        why = _blocked(ipaddress.ip_address(info[4][0]), private_ok)
        if why:
            raise CardFetchError(400, f"agent_url points to a {why} address")


def pick_card(data: Any) -> Dict[str, Any]:
    """카드 모양인지 보고 카드 필드만 남긴다 — 다른 JSON 을 그대로 돌려주지 않는다."""
    if not isinstance(data, dict) or not isinstance(data.get("name"), str) or not data["name"].strip():
        raise CardFetchError(502, "response is not an A2A agent card")
    card = {k: data[k] for k in CARD_FIELDS if k in data}
    skills = data.get("skills")
    if isinstance(skills, list):
        card["skills"] = [{k: s[k] for k in SKILL_FIELDS if k in s} for s in skills if isinstance(s, dict)]
    else:
        card.pop("skills", None)
    return card


def _get(url: str) -> requests.Response:
    parts = urlsplit(url)
    check_host(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    # 넘겨주기(redirect)는 따라가지 않는다 — 넘겨받은 곳이 내부 주소일 수 있다
    return requests.get(url, timeout=TIMEOUT, allow_redirects=False, stream=True,
                        headers={"Accept": "application/json"})


def fetch_agent_card(agent_url: str) -> Dict[str, Any]:
    last: Optional[CardFetchError] = None
    for url in card_urls(agent_url):
        try:
            resp = _get(url)
        except requests.RequestException as e:
            raise CardFetchError(502, f"cannot reach agent: {type(e).__name__}")
        with resp:
            if resp.status_code == 404:
                last = CardFetchError(404, f"no agent card at {urlsplit(url).path}")
                continue
            if 300 <= resp.status_code < 400:
                raise CardFetchError(502, f"agent card request was redirected ({resp.status_code}); use the final URL")
            if resp.status_code != 200:
                raise CardFetchError(502, f"agent returned HTTP {resp.status_code}")
            body = resp.raw.read(MAX_BYTES + 1, decode_content=True)
            if len(body) > MAX_BYTES:
                raise CardFetchError(502, "agent card is too large")
            try:
                data = json.loads(body)
            except ValueError:
                raise CardFetchError(502, "agent card is not JSON")
            return pick_card(data)
    raise last or CardFetchError(404, "no agent card found")
