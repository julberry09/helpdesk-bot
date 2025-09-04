# tests/test_api.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import re
import pytest
from fastapi.testclient import TestClient

from helpdesk_bot.api import api
from helpdesk_bot import core as core_logic


# =============================================================
# Fixtures & Helpers
# =============================================================
@pytest.fixture(scope="module")
def client():
    """
    FastAPI 테스트 클라이언트를 반환합니다.
    테스트 클라이언트는 API 엔드포인트 테스트를 위해 사용됩니다.
    """
    return TestClient(api)

def force_recompile_graph():
    """
    LangGraph의 `_graph` 속성을 초기화합니다.
    노드 수정 후 강제로 재컴파일을 진행하는 데 유용합니다.
    """
    core_logic._graph = None

# =============================================================
# 1. API 기본 동작 테스트
# =============================================================
def run_api_test(client, endpoint, payload, expected_status, expected_keys=None, mock_functions=None, monkeypatch=None, additional_assertions=None):
    """
    API 테스트를 위한 유틸리티 함수
    
    Args:
        client: TestClient 인스턴스
        endpoint: 테스트할 API 엔드포인트
        payload: 요청에 사용할 JSON 페이로드
        expected_status: 예상하는 HTTP 상태 코드
        expected_keys: 응답 JSON에 포함되어야 하는 키 목록 (Optional)
        mock_functions: [(원본 함수 경로, 가짜 함수), ...] 형태의 리스트 (Optional)
        monkeypatch: pytest의 monkeypatch 픽스처
        additional_assertions: 추가적인 커스텀 검증 함수 (Optional)
    """
    # 1. 목(mock) 함수 설정
    if mock_functions and monkeypatch:
        for module_path_str, func in mock_functions:
            monkeypatch.setattr(module_path_str, func)
            
    # 2. API 요청 실행
    response = client.post(endpoint, json=payload)
    
    # 3. 상태 코드 검증
    assert response.status_code == expected_status
    
    # 4. 응답 키 검증 (선택적)
    data = response.json()
    if expected_keys:
        for key in expected_keys:
            assert key in data, f"응답에 {key} 키가 누락되었습니다."

    # 추가 검증이 있다면 실행
    if additional_assertions:
        additional_assertions(data)

    return data

def test_health_ok(client):
    """/health 엔드포인트가 정상적으로 200 OK와 {"ok": True}를 반환하는지 테스트합니다."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

def test_chat_bad_request(client):
    """'message' 필드가 없는 잘못된 요청에 대해 422 Unprocessable Entity를 반환하는지 테스트합니다."""
    r = client.post("/chat", json={"session_id": "bad_request"})
    assert r.status_code == 422

# =============================================================
# 2. 인텐트별 상세 로직 테스트
# =============================================================
def test_reset_password_flow(client, monkeypatch):
    """'reset_password' 인텐트 흐름을 테스트합니다."""
    def fake_classify(state):
        return {"intent": "reset_password", "tool_output": {"user": "kim.s"}}

    def assert_reset_pw_response(data):
        assert data["intent"] == "reset_password"
        assert "비밀번호 초기화 안내" in data["reply"]
        assert "본인인증" in data["reply"]

    run_api_test(
        client,
        endpoint="/chat",
        payload={"message": "비밀번호 초기화 해줘"},
        expected_status=200,
        expected_keys=["reply", "intent"],
        mock_functions=[("src.helpdesk_bot.core.node_classify", fake_classify)],
        monkeypatch=monkeypatch,
        additional_assertions=assert_reset_pw_response
    )

def test_request_id_flow(client, monkeypatch):
    """'request_id' 인텐트 흐름을 테스트합니다."""
    def fake_classify(state):
        state["intent"] = "request_id"
        state["tool_output"] = {"name": "홍길동", "dept": "IT운영"}
        return state

    def assert_request_id_response(data):
        assert data["intent"] == "request_id"
        assert "ID 발급 신청" in data["reply"]
        assert re.search(r"REQ-\d+", data["reply"])

    run_api_test(
        client,
        endpoint="/chat",
        payload={"message": "신규 계정 발급 신청"},
        expected_status=200,
        expected_keys=["reply", "intent"],
        mock_functions=[("src.helpdesk_bot.core.node_classify", fake_classify)],
        monkeypatch=monkeypatch,
        additional_assertions=assert_request_id_response
    )

def test_owner_lookup_flow(client, monkeypatch):
    """'owner_lookup' 인텐트 흐름을 테스트합니다."""
    def fake_classify(state):
        state["intent"] = "owner_lookup"
        state["tool_output"] = {"screen": "인사시스템-사용자관리"}
        return state

    def assert_owner_lookup_response(data):
        assert data["intent"] == "owner_lookup"
        assert "담당자" in data["reply"]

    run_api_test(
        client,
        endpoint="/chat",
        payload={"message": "인사시스템 사용자관리 담당자"},
        expected_status=200,
        expected_keys=["reply", "intent"],
        mock_functions=[("src.helpdesk_bot.core.node_classify", fake_classify)],
        monkeypatch=monkeypatch,
        additional_assertions=assert_owner_lookup_response
    )

def test_rag_flow_with_fake_answer(client, monkeypatch):
    """'rag_qa' 인텐트 흐름을 테스트합니다."""
    def fake_classify(state):
        state["intent"] = "rag_qa"
        state["tool_output"] = {}
        return state
        
    def fake_rag(state):
        state["result"] = "핵심 요약: 사내 규정에 따라 신청 양식을 제출해야 합니다."
        state["sources"] = [{"index": 1, "source": "seed-faq.txt", "page": None}]
        return state

    def assert_rag_response(data):
        assert data["intent"] == "rag_qa"
        assert "핵심 요약" in data["reply"]
        assert isinstance(data.get("sources"), list) and len(data["sources"]) >= 1

    run_api_test(
        client,
        endpoint="/chat",
        payload={"message": "ID 발급 절차 알려줘"},
        expected_status=200,
        expected_keys=["reply", "intent", "sources"],
        mock_functions=[
            ("src.helpdesk_bot.core.node_classify", fake_classify),
            ("src.helpdesk_bot.core.node_rag", fake_rag)
        ],
        monkeypatch=monkeypatch,
        additional_assertions=assert_rag_response
    )