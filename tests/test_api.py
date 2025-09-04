# tests/test_api.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import re
import pytest
from fastapi.testclient import TestClient

from helpdesk_bot.api import api
from helpdesk_bot.core import AZURE_AVAILABLE, build_or_load_vectorstore


# =============================================================
# Fixtures & Helpers
# =============================================================
@pytest.fixture(scope="module")
def client():
    """
    FastAPI 테스트 클라이언트를 반환합니다.
    """
    return TestClient(api)

@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """
    테스트에 필요한 환경을 설정합니다.
    (예: 테스트용 벡터스토어 재빌드)
    """
    if AZURE_AVAILABLE:
        print("\nAZURE_AVAILABLE: 테스트를 위해 벡터스토어를 재빌드합니다.")
        try:
            build_or_load_vectorstore()
        except RuntimeError as e:
            pytest.skip(f"Azure OpenAI 설정이 없어 테스트를 건너뜁니다: {e}")
    else:
        print("\nAZURE_AVAILABLE: False, 폴백 모드로 테스트합니다.")
    yield

def run_api_test(client, endpoint, payload, expected_status, expected_keys=None, additional_assertions=None):
    """
    API 테스트를 위한 유틸리티 함수
    
    Args:
        client: TestClient 인스턴스
        endpoint: 테스트할 API 엔드포인트
        payload: 요청에 사용할 JSON 페이로드
        expected_status: 예상하는 HTTP 상태 코드
        expected_keys: 응답 JSON에 포함되어야 하는 키 목록 (Optional)
        additional_assertions: 추가적인 커스텀 검증 함수 (Optional)
    """
    # 1. API 요청 실행
    response = client.post(endpoint, json=payload)
    
    # 2. 상태 코드 검증
    assert response.status_code == expected_status
    
    # 3. 응답 키 검증 (선택적)
    data = response.json()
    if expected_keys:
        for key in expected_keys:
            assert key in data, f"응답에 {key} 키가 누락되었습니다."

    # 4. 추가 검증이 있다면 실행
    if additional_assertions:
        additional_assertions(data, response)

    return data

# =============================================================
# 1. API 기본 동작 테스트
# =============================================================
def test_health_ok(client):
    """/health 엔드포인트가 정상적으로 200 OK와 {"ok": True}를 반환하는지 테스트합니다."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

def test_chat_bad_request(client):
    """'message' 필드가 없는 잘못된 요청에 대해 422 Unprocessable Entity를 반환하는지 테스트합니다."""
    r = client.post("/chat", json={"session_id": "bad_request"})
    assert r.status_code == 422
    assert "message" in r.json()["detail"][0]["loc"]

# =============================================================
# 2. 통합 테스트: 전체 파이프라인 동작 검증
# =============================================================

def test_rag_flow_integration(client):
    """
    RAG 파이프라인의 전체 동작을 통합 테스트합니다.
    """
    def assert_rag_response(data, response):
        assert data["intent"] in ["rag_qa", "fallback_no_match"]

        if AZURE_AVAILABLE:
            assert "rag_qa" in data["intent"]
            assert "ID 발급" in data["reply"]
            assert isinstance(data.get("sources"), list)
            assert len(data["sources"]) > 0
        else:
            assert "fallback_no_match" in data["intent"]
            assert "복잡한 질문에 답변할 수 없습니다" in data["reply"]
            assert len(data["sources"]) == 0

    run_api_test(
        client,
        endpoint="/chat",
        payload={"message": "ID 발급 절차 알려줘"},
        expected_status=200,
        expected_keys=["reply", "intent", "sources"],
        additional_assertions=assert_rag_response
    )

def test_tool_owner_lookup_integration(client):
    """
    '담당자 조회' 도구 호출 흐름을 통합 테스트합니다.
    """
    def assert_owner_lookup_response(data, response):
        assert data["intent"] == "owner_lookup"
        assert "담당자" in data["reply"]
        assert "홍길동" in data["reply"]
    
    run_api_test(
        client,
        endpoint="/chat",
        payload={"message": "인사시스템 사용자관리 담당자 알려줘"},
        expected_status=200,
        expected_keys=["reply", "intent"],
        additional_assertions=assert_owner_lookup_response
    )

def test_tool_reset_password_integration(client):
    """
    '비밀번호 초기화' 도구 호출 흐름을 통합 테스트합니다.
    """
    def assert_reset_pw_response(data, response):
        assert data["intent"] == "reset_password"
        assert "비밀번호 초기화 안내" in data["reply"]

    run_api_test(
        client,
        endpoint="/chat",
        payload={"message": "비밀번호 초기화"},
        expected_status=200,
        expected_keys=["reply", "intent"],
        additional_assertions=assert_reset_pw_response
    )

def test_tool_request_id_integration(client):
    """
    'ID 발급 신청' 도구 호출 흐름을 통합 테스트합니다.
    """
    def assert_request_id_response(data, response):
        assert data["intent"] == "request_id"
        assert "ID 발급 신청" in data["reply"]
        assert re.search(r"REQ-\d+", data["reply"])
    
    run_api_test(
        client,
        endpoint="/chat",
        payload={"message": "계정 발급 신청"},
        expected_status=200,
        expected_keys=["reply", "intent"],
        additional_assertions=assert_request_id_response
    )