# tests/test_api.py
import json
import re
import pytest
from fastapi.testclient import TestClient

# [수정 1] 새로운 파일 구조에 맞게 import 변경
# - api.py에서 FastAPI 앱을 가져옴
# - core.py에서 핵심 로직/노드를 가져옴 (별칭 사용)
from helpdesk_bot.api import api
from helpdesk_bot import core as core_logic

# =============================================================
# Fixtures & Helpers
# =============================================================
@pytest.fixture(scope="module")
def client():
    """FastAPI 테스트 클라이언트 Fixture"""
    # [수정 2] TestClient에 새로 import한 api 객체를 전달
    return TestClient(api)

def force_recompile_graph():
    """
    노드를 monkeypatch한 후 LangGraph가 다시 빌드되도록 강제합니다.
    """
    # [수정 3] appmod._graph 대신 core_logic._graph를 참조
    core_logic._graph = None

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

# =============================================================
# 2. 인텐트별 상세 로직 테스트
# =============================================================
def test_reset_password_flow(client, monkeypatch):
    """'reset_password' 인텐트 흐름을 테스트합니다."""
    def fake_classify(state):
        state["intent"] = "reset_password"
        state["tool_output"] = {"user": "kim.s"}
        return state

    # [수정 4] monkeypatch 대상을 appmod에서 core_logic으로 변경
    monkeypatch.setattr(core_logic, "node_classify", fake_classify, raising=True)
    force_recompile_graph()

    r = client.post("/chat", json={"message": "비밀번호 초기화 해줘"})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "reset_password"
    assert "비밀번호 초기화" in data["reply"]

def test_request_id_flow(client, monkeypatch):
    """'request_id' 인텐트 흐름을 테스트합니다."""
    def fake_classify(state):
        state["intent"] = "request_id"
        state["tool_output"] = {"name": "홍길동", "dept": "IT운영"}
        return state

    # [수정 4] monkeypatch 대상을 appmod에서 core_logic으로 변경
    monkeypatch.setattr(core_logic, "node_classify", fake_classify, raising=True)
    force_recompile_graph()

    r = client.post("/chat", json={"message": "신규 계정 발급 신청"})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "request_id"
    assert "ID 발급 신청" in data["reply"]
    assert re.search(r"REQ-\d+", data["reply"])

def test_owner_lookup_flow(client, monkeypatch):
    """'owner_lookup' 인텐트 흐름을 테스트합니다."""
    def fake_classify(state):
        state["intent"] = "owner_lookup"
        state["tool_output"] = {"screen": "인사시스템-사용자관리"}
        return state

    # [수정 4] monkeypatch 대상을 appmod에서 core_logic으로 변경
    monkeypatch.setattr(core_logic, "node_classify", fake_classify, raising=True)
    force_recompile_graph()

    r = client.post("/chat", json={"message": "인사시스템 사용자관리 담당자"})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "owner_lookup"
    assert "담당자" in data["reply"]

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

    # [수정 4] monkeypatch 대상을 appmod에서 core_logic으로 변경
    monkeypatch.setattr(core_logic, "node_classify", fake_classify, raising=True)
    monkeypatch.setattr(core_logic, "node_rag", fake_rag, raising=True)
    force_recompile_graph()

    r = client.post("/chat", json={"message": "ID 발급 절차 알려줘"})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "rag_qa"
    assert "핵심 요약" in data["reply"]
    assert isinstance(data.get("sources"), list) and len(data["sources"]) >= 1

    
# =============================================================
# import types # 파이썬 내부 객체 타입 관련 모듈 (FunctionType, GeneratorType, ModuleType 등 제공)
# 3. 통합 테스트: 전체 파이프라인 동작 테스트 (End-to-End Pipeline Tests)
# def test_chat_with_monkeypatched_pipeline(client, monkeypatch):
#     """
#     외부 LLM/AOAI 호출 없이도 테스트가 가능하도록 pipeline을 스텁으로 교체.
#     """
#     def fake_pipeline(question: str):
#         # 간단한 라우팅 흉내 + 결과 스텁
#         if "비밀번호" in question:
#             return {
#                 "intent": "reset_password",
#                 "result": "✅ 비밀번호 초기화 안내\n\n1. SSO 포털 접속 > 비밀번호 재설정\n2. 본인인증\n3. 새 비밀번호 설정",
#                 "sources": [],
#                 "tool_output": {"ok": True},
#             }
#         elif "담당자" in question:
#             return {
#                 "intent": "owner_lookup",
#                 "result": "👤 '인사시스템-사용자관리' 담당자\n- 이름: 홍길동\n- 이메일: owner.hr@example.com",
#                 "sources": [],
#                 "tool_output": {"ok": True},
#             }
#         else:
#             return {"intent": "rag_qa", "result": "일반 안내입니다.", "sources": [], "tool_output": {}}

#     # pipeline monkeypatch
#     monkeypatch.setattr(appmod, "pipeline", fake_pipeline)

#     payload = {"message": "비밀번호 초기화 방법 알려줘"}
#     res = client.post("/chat", json=payload)
#     assert res.status_code == 200
#     data = res.json()
#     assert data["intent"] == "reset_password"
#     assert "비밀번호" in data["reply"]


# def test_pipeline_smoke(monkeypatch):
#     """
#     pipeline 자체를 직접 호출해 스모크 테스트 (LLM 호출은 스킵).
#     LangGraph 내부 invoke를 더미 함수로 바꿔 최소 동작만 검증.
#     """
#     class FakeCompiledGraph:
#         def invoke(self, state):
#             return {"intent": "rag_qa", "result": "스모크 테스트 OK", "sources": []}

#     def fake_build_graph():
#         return FakeCompiledGraph()

#     # 그래프/LLM 호출 우회
#     monkeypatch.setattr(appmod, "build_graph", fake_build_graph)
#     # 전역 그래프 초기화
#     if hasattr(appmod, "_graph"):
#         appmod._graph = None

#     out = appmod.pipeline("테스트 질문")
#     assert out["result"] == "스모크 테스트 OK"