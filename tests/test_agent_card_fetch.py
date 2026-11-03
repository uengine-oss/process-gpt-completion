"""
A2A 에이전트 카드 가져오기(`/multi-agent/fetch-data`) 검증.

예전 구현은 받은 주소 뒤에 `/.well-known/agent.json` 을 이어 붙여 아무 데나 GET 하고 그 JSON 을 그대로 돌려줬다.
그래서 질의 문자열로 경로를 삼키면 내부의 아무 JSON 이나, 169.254.169.254 면 클라우드 자격 증명까지 읽을 수 있었다.
여기서 지키려는 것은 (1) 카드 주소만 만든다, (2) 닿으면 안 되는 주소는 이름을 풀어서라도 막는다,
(3) 카드 모양인 응답의 카드 필드만 돌려준다, (4) 새 카드 위치(agent-card.json)와 옛 위치를 모두 찾는다.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent_card_fetch as acf  # noqa: E402
from agent_card_fetch import CardFetchError, card_urls, check_host, fetch_agent_card, pick_card  # noqa: E402


def test_card_urls_tries_new_then_old_location():
    assert card_urls("agent.example.com:9999") == [
        "http://agent.example.com:9999/.well-known/agent-card.json",
        "http://agent.example.com:9999/.well-known/agent.json",
    ]


def test_card_urls_keeps_path_prefix_and_falls_back_to_root():
    assert card_urls("https://host/a2a/sales/") == [
        "https://host/a2a/sales/.well-known/agent-card.json",
        "https://host/a2a/sales/.well-known/agent.json",
        "https://host/.well-known/agent-card.json",
        "https://host/.well-known/agent.json",
    ]


def test_card_url_given_directly_is_used_as_is():
    assert card_urls("https://host/.well-known/agent.json") == ["https://host/.well-known/agent.json"]


@pytest.mark.parametrize("bad", [
    "http://internal/secret?x=",          # 뒤에 붙는 카드 경로를 질의로 삼키던 길
    "http://internal/secret#x",
    "http://user:pw@host",
    "file:///etc/passwd",
    "gopher://host",
    "http://",
    "http://host:99999",
    "",
])
def test_card_urls_rejects(bad):
    with pytest.raises(CardFetchError) as e:
        card_urls(bad)
    assert e.value.status == 400


def _resolve_to(monkeypatch, *ips):
    monkeypatch.setattr(acf.socket, "getaddrinfo",
                        lambda host, port, type=0: [(socket.AF_INET6 if ":" in ip else socket.AF_INET, type, 6, "", (ip, port))
                                                    for ip in ips])


@pytest.mark.parametrize("ip", ["127.0.0.1", "0.0.0.0", "::1", "169.254.169.254", "fe80::1", "::ffff:127.0.0.1", "224.0.0.1"])
def test_check_host_blocks_loopback_metadata_and_reserved(monkeypatch, ip):
    _resolve_to(monkeypatch, ip)
    with pytest.raises(CardFetchError):
        check_host("looks-harmless.example.com", 80)


def test_check_host_blocks_if_any_address_is_bad(monkeypatch):
    _resolve_to(monkeypatch, "93.184.216.34", "169.254.169.254")
    with pytest.raises(CardFetchError):
        check_host("two-faced.example.com", 80)


def test_private_network_allowed_by_default_and_blockable(monkeypatch):
    _resolve_to(monkeypatch, "172.18.0.5")
    monkeypatch.delenv("A2A_FETCH_ALLOW_PRIVATE", raising=False)
    check_host("a2a-agent", 9999)  # 설치본: 같은 도커 망의 에이전트
    monkeypatch.setenv("A2A_FETCH_ALLOW_PRIVATE", "false")
    with pytest.raises(CardFetchError):
        check_host("a2a-agent", 9999)


def test_public_address_passes(monkeypatch):
    _resolve_to(monkeypatch, "93.184.216.34")
    check_host("agent.example.com", 443)


def test_pick_card_keeps_card_fields_only():
    card = pick_card({
        "name": "견적 에이전트", "description": "견적을 냅니다", "url": "http://x", "secret": "s3cr3t",
        "skills": [{"id": "quote", "name": "견적", "internal": 1}, "junk"],
    })
    assert card == {"name": "견적 에이전트", "description": "견적을 냅니다", "url": "http://x",
                    "skills": [{"id": "quote", "name": "견적"}]}


@pytest.mark.parametrize("not_card", [[], {"token": "x"}, {"name": ""}, {"name": 3}, "text"])
def test_pick_card_rejects_non_cards(not_card):
    with pytest.raises(CardFetchError) as e:
        pick_card(not_card)
    assert e.value.status == 502


class _Agent(BaseHTTPRequestHandler):
    routes: dict = {}

    def do_GET(self):  # noqa: N802
        status, body, headers = self.routes.get(self.path, (404, b"", {}))
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def agent(monkeypatch):
    """127.0.0.1 에 뜬 가짜 A2A 에이전트. 주소 검사만 건너뛰고 나머지는 실제 요청으로 본다."""
    monkeypatch.setattr(acf, "check_host", lambda host, port: None)
    server = HTTPServer(("127.0.0.1", 0), _Agent)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Agent.routes = {}
    yield _Agent.routes, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _json(obj, status=200):
    return status, json.dumps(obj).encode(), {"Content-Type": "application/json"}


def test_fetch_new_card_location(agent):
    routes, base = agent
    routes["/.well-known/agent-card.json"] = _json({"name": "A", "skills": [{"id": "s"}], "extra": 1})
    assert fetch_agent_card(base) == {"name": "A", "skills": [{"id": "s"}]}


def test_fetch_falls_back_to_old_card_location(agent):
    routes, base = agent
    routes["/.well-known/agent.json"] = _json({"name": "Old"})
    assert fetch_agent_card(base)["name"] == "Old"


def test_fetch_reports_missing_card(agent):
    _, base = agent
    with pytest.raises(CardFetchError) as e:
        fetch_agent_card(base)
    assert e.value.status == 404


def test_fetch_does_not_follow_redirects(agent):
    routes, base = agent
    routes["/.well-known/agent-card.json"] = (302, b"", {"Location": "http://169.254.169.254/latest/meta-data/"})
    with pytest.raises(CardFetchError) as e:
        fetch_agent_card(base)
    assert "redirected" in str(e.value)


def test_fetch_checks_status(agent):
    routes, base = agent
    routes["/.well-known/agent-card.json"] = _json({"name": "A"}, status=500)
    with pytest.raises(CardFetchError) as e:
        fetch_agent_card(base)
    assert e.value.status == 502


def test_fetch_rejects_oversized_and_non_json(agent):
    routes, base = agent
    routes["/.well-known/agent-card.json"] = (200, b"{" + b" " * (acf.MAX_BYTES + 10), {})
    with pytest.raises(CardFetchError, match="too large"):
        fetch_agent_card(base)
    routes["/.well-known/agent-card.json"] = (200, b"<html>", {})
    with pytest.raises(CardFetchError, match="not JSON"):
        fetch_agent_card(base)
