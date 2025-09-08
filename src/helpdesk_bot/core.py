# src/helpdesk_bot/core.py

import os
import json
import logging
import csv
import time as _time
from typing import TypedDict, List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from konlpy.tag import Okt

from . import constants

# =============================================================
# 1. 공통 설정 / 환경 변수
# =============================================================
load_dotenv()

# 로거 설정
logger = logging.getLogger("helpdesk-bot")
if not logger.handlers:
    LOG_DIR = Path("./logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
    class _ConsoleFormatter(logging.Formatter):
        def format(self, record):
            base = {"level": record.levelname, "name": record.name, "msg": record.getMessage()}
            if hasattr(record, "extra_data"):
                base.update(record.extra_data)
            return json.dumps(base, ensure_ascii=False)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(_ConsoleFormatter())
    file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    file_handler.setFormatter(_ConsoleFormatter())
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Azure OpenAI 환경변수
AOAI_ENDPOINT    = os.getenv("AOAI_ENDPOINT", "")
AOAI_API_KEY     = os.getenv("AOAI_API_KEY", "")
AOAI_API_VERSION = os.getenv("AOAI_API_VERSION", "2024-10-21")
AOAI_DEPLOY_GPT4O_MINI = os.getenv("AOAI_DEPLOY_GPT4O_MINI", "gpt-4o-mini")
AOAI_DEPLOY_GPT4O = os.getenv("AOAI_DEPLOY_GPT4O", "gpt-4o")
AOAI_DEPLOY_EMBED_3_SMALL = os.getenv("AOAI_DEPLOY_EMBED_3_SMALL", "text-embedding-3-small")

# Azure 설정 확인 플래그
AZURE_AVAILABLE = bool(AOAI_ENDPOINT and AOAI_API_KEY)
if not AZURE_AVAILABLE:
    logger.warning("Azure OpenAI 설정이 없어 폴백(Fallback) 모드로 동작합니다.")

# Okt 형태소 분석기 초기화
okt = Okt()

# =============================================================
# 4. Fallback & Main Pipelines 
# =============================================================
# 임베딩 모델 생성
def _make_embedder() -> AzureOpenAIEmbeddings:
    if not AZURE_AVAILABLE:
        raise RuntimeError("Azure OpenAI 설정이 없어 Embedder를 생성할 수 없습니다.")
    return AzureOpenAIEmbeddings(
        azure_deployment=AOAI_DEPLOY_EMBED_3_SMALL,
        api_key=AOAI_API_KEY,
        azure_endpoint=AOAI_ENDPOINT,
        api_version=AOAI_API_VERSION,
    
# RAG - 원본 데이터 수집 및 전처리 로직 [checklist: 6] 
def _load_docs_from_kb() -> List[Document]:
    docs: List[Document] = []
    for kb_path in [KB_DEFAULT_DIR, KB_DATA_DIR]:
        if not kb_path.exists():
            kb_path.mkdir(parents=True, exist_ok=True)
        for p in kb_path.rglob("*"):
            if p.is_file():
                try:
                    suf = p.suffix.lower()
                    if suf == ".pdf": docs.extend(PyPDFLoader(str(p)).load())
                    elif suf == ".csv" and p.name != "faq_data.csv":
                        docs.extend(CSVLoader(file_path=str(p), encoding="utf-8").load())
                    elif suf in [".txt", ".md"]: docs.extend(TextLoader(str(p), encoding="utf-8").load())
                    elif suf == ".docx": docs.extend(Docx2txtLoader(str(p)).load())
                except Exception as e:
                    logger.warning(f"문서 로드 실패: {p} - {e}")
    return docs

# RAG - FAISS 기반의 Vector 스토어 구축 [checklist: 7] 
def build_or_load_vectorstore() -> FAISS:
    if not AZURE_AVAILABLE:
        raise RuntimeError("'Rebuild Index'는 Azure OpenAI 설정이 필요합니다.")
        
    embed = _make_embedder()
    if (INDEX_DIR / f"{INDEX_NAME}.faiss").exists():
        return FAISS.load_local(str(INDEX_DIR / INDEX_NAME), embeddings=embed, allow_dangerous_deserialization=True)

    raw_docs = _load_docs_from_kb()
    
    if not raw_docs:
        faq_data = load_faq_data()
        if faq_data:
            raw_docs = [
                Document(
                    page_content=f"질문: {item.get('question')}\n답변: {item.get('answer')}",
                    metadata={"source": "faq_data.csv"}
                ) for item in faq_data
            ]
            logger.info("업로드된 문서가 없어 faq_data.csv를 기본 RAG 지식으로 사용합니다.")
        else:
            seed_text = """사내 헬프데스크 안내
- ID 발급: 신규 입사자는 HR 포털에서 '계정 신청' 양식을 제출. 승인 후 IT가 계정 생성.
- 비밀번호 초기화: SSO 포털의 '비밀번호 재설정' 기능 사용. 본인인증 필요.
- 담당자 조회: 포털 상단 검색창에 화면/메뉴명을 입력하면 담당자 카드가 표시됨."""
            raw_docs = [Document(page_content=seed_text, metadata={"source": "seed-faq.txt"})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(raw_docs)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # FAISS에 문서를 임베딩하고 저장
    vs = FAISS.from_documents(chunks, embed)
    vs.save_local(str(INDEX_DIR / INDEX_NAME))
    return vs

# RAG - FAISS 벡터 스토어 검색기 (Singleton Pattern)
_vectorstore: Optional[FAISS] = None
def retriever(k: int = 4):
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_or_load_vectorstore()
    return _vectorstore.as_retriever(search_kwargs={"k": k})

# LLM(언어 모델) 인스턴스를 생성
def make_llm(model: str = AOAI_DEPLOY_GPT4O_MINI, temperature: float = 0.2) -> AzureChatOpenAI:
    """
    Azure OpenAI 서비스에 연결하여 LLM(언어 모델) 인스턴스를 생성합니다.
    Args:
        model (str): 사용할 Azure OpenAI 배포 모델의 이름. 기본값은 gpt-4o-mini입니다.
        temperature (float): 모델의 창의성(무작위성)을 조절하는 매개변수. 0.0에서 2.0 사이의 값. 
                           값이 낮을수록 예측 가능하고 일관된 답변을 생성합니다.
    Returns:
        AzureChatOpenAI: 설정된 언어 모델 인스턴스.
    Raises:
        RuntimeError: Azure OpenAI 환경 변수(엔드포인트, API 키)가 설정되지 않은 경우 발생.
    """
    if not AZURE_AVAILABLE:
        raise RuntimeError("Azure OpenAI 설정이 없어 LLM을 생성할 수 없습니다.")
    return AzureChatOpenAI(
        azure_deployment=model,
        api_version=AOAI_API_VERSION,
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        temperature=temperature,
    )
# =============================================================
# 3. LangGraph 도구 및 노드 정의
# ==========================================================
# 상태 관리 (State Management)
class BotState(TypedDict):
    question: str; intent: str; result: str
    sources: List[Dict[str, Any]]; tool_output: Dict[str, Any]
# 도구(Tool) 함수
def tool_reset_password(payload: Dict[str, Any]) -> Dict[str, Any]:
    """비밀번호 초기화 절차를 안내합니다."""
    return {
        "ok": True, 
        "message": "비밀번호 초기화 절차 안내", 
        "steps": ["SSO 포털 접속 > 비밀번호 재설정", "본인인증", "새 비밀번호 설정"]
    }

def tool_request_id(payload: Dict[str, Any]) -> Dict[str, Any]:
    """ID 발급 신청 절차를 안내합니다."""
    return {
        "ok": True, 
        "message": "ID 발급 신청 절차 안내", 
        "steps": ["HR 포털 접속 > '계정 신청' 양식 제출", "양식 승인 후 IT팀에서 계정 생성"]
    }

def tool_owner_lookup(payload: Dict[str, Any]) -> Dict[str, Any]:
    screen = payload.get("screen") or ""
    info = OWNER_FALLBACK.get(screen)
    if not info:
        return {"ok": False, "message": f"'{screen}' 담당자 정보를 찾지 못했습니다."}
    return {"ok": True, "screen": screen, "owner": info}

# 노드(Node) 함수
# Prompt Engineering - 사용자 의도 분석, 다양한 질문에 일관된 응답을 도출하도록 설계 (프롬프트 재사용성) [checklist: 2]
def node_classify(state: BotState) -> BotState:
    llm = make_llm()
    sys_prompt = ("당신은 사내 헬프데스크 라우터입니다. 사용자 입력을 reset_password, request_id, owner_lookup, rag_qa 중 하나로 분류하세요. JSON(intent, arguments)으로만 답하세요.")
    msg = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": state["question"]}]
    out = llm.invoke(msg).content
    intent, args = "rag_qa", {}
    try:
        data = json.loads(out)
        intent = data.get("intent", "rag_qa")
        args = data.get("arguments", {}) or {}
    except json.JSONDecodeError:
        logger.warning(f"[Supervisor JSON 오류] JSONDecodeError: {out}")
    except Exception:
        logger.error(f"[Supervisor 오류] 알 수 없는 오류: {out}")
    return {**state, "intent": intent, "tool_output": args}

def node_reset_pw(state: BotState) -> BotState: return {**state, "tool_output": tool_reset_password(state.get("tool_output", {}))}

def node_request_id(state: BotState) -> BotState: return {**state, "tool_output": tool_request_id(state.get("tool_output", {}))}

def node_owner_lookup(state: BotState) -> BotState: return {**state, "tool_output": tool_owner_lookup(state.get("tool_output", {}))}

# RAG - 사전 정의된 데이터(문서)를 검색하여 AI의 논리력을 보강/ RAG 기반 지식 검색 기능 구현 [checklist: 8,9] 
# Prompt Engineering - 프롬프트 최적화 (역할 부여 + Chain-of-Thought) [checklist: 1] 
def node_rag(state: BotState) -> BotState:
    docs = retriever(k=4).get_relevant_documents(state["question"])
    context = "\n\n".join([f"[{i+1}] {d.page_content[:1200]}" for i, d in enumerate(docs)])
    sources = [{"index": i+1, "source": d.metadata.get("source","unknown"), "page": d.metadata.get("page")} for i,d in enumerate(docs)]
    llm = make_llm(model=AOAI_DEPLOY_GPT4O)
    sys_prompt = "너는 사내 헬프데스크 상담원이다. 컨텍스트를 기반으로 실행 가능한 답변을 한국어로 작성해라."
    user_prompt = f"질문:\n{state['question']}\n\n컨텍스트:\n{context}"
    out = llm.invoke([{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}]).content
    return {**state, "result": out, "sources": sources}

# 도구(Tool) 함수 결과 사용자 친화적인 형태로 변환
def node_finalize(state: BotState) -> BotState:
    if state["intent"] in ["reset_password", "request_id", "owner_lookup"]:
        res = state.get("tool_output", {})
        if state["intent"] == "reset_password":
            text = f"✅ 비밀번호 초기화 안내\n\n" + "\n".join(f"{i+1}. {s}" for i,s in enumerate(res.get("steps", []))) if res.get("ok") else f"❗{res.get('message','실패')}"
        elif state["intent"] == "request_id":
            text = f"🆔 ID 발급 신청\n상태: {'접수됨' if res.get('ok') else '실패'}\n티켓: {res.get('ticket','-')}"
        else:
            text = f"👤 '{res.get('screen')}' 담당자\n- 이름: {res.get('owner', {}).get('owner')}\n- 이메일: {res.get('owner', {}).get('email')}\n- 연락처: {res.get('owner', {}).get('phone')}" if res.get("ok") else f"❗{res.get('message','조회 실패')}"
        return {**state, "result": text}
    return state

# LangChain & LangGraph - Multi Agent 형태의 Agent Flow 설계 및 구현/ ReAct (Reasoning and Acting) 사용/ 멀티턴 대화 (memory) [checklist: 3,4,5]
_memory_checkpointer = MemorySaver()
_graph = None
def build_graph():
    # StateGraph 클래스를 사용해 멀티 에이전트 워크플로우를 정의함
    g = StateGraph(BotState)
    g.add_node("classify", node_classify)
    g.add_node("reset_password", node_reset_pw)
    g.add_node("request_id", node_request_id)
    g.add_node("owner_lookup", node_owner_lookup)
    g.add_node("rag", node_rag)
    g.add_node("finalize", node_finalize)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", lambda s: s["intent"], {"reset_password":"finalize", "request_id":"finalize", "owner_lookup":"finalize", "rag":"rag"})
    g.add_edge("finalize", END); g.add_edge("rag", END)
    return g.compile(checkpointer=_memory_checkpointer)

# =============================================================
# 4. Fallback & Main Pipelines
# =============================================================
_faq_data = None
def load_faq_data() -> List[Dict[str, str]]:
    """kb_default/faq_data.csv 파일을 읽어 메모리에 로드합니다."""
    global _faq_data
    if _faq_data is not None:
        return _faq_data
    
    faq_file_path = KB_DEFAULT_DIR / "faq_data.csv"
    if not faq_file_path.exists():
        _faq_data = []
        return _faq_data
    
    loaded_data = []
    try:
        with open(faq_file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # FAQ 질문을 미리 형태소 분석하여 저장
                row["faq_words"] = set(okt.phrases(row.get("question", "")))
                loaded_data.append(row)
        logger.info(f"{len(loaded_data)}개의 FAQ 데이터를 로드했습니다.")
    except Exception as e:
        logger.error(f"FAQ 파일 로드 실패: {e}")
    
    _faq_data = loaded_data
    return _faq_data

def find_similar_faq(question: str) -> Optional[str]:
    """자카드 유사도를 사용해 가장 비슷한 FAQ 질문을 찾고 답변을 반환합니다."""
    faq_data = load_faq_data()
    if not faq_data:
        return None

    # 사용자 질문을 형태소 분석
    user_words = set(okt.phrases(question.lower()))
    
    # 만약 phrases()가 제대로 된 키워드를 추출하지 못할 경우를 대비하여
    # 명사 추출을 추가로 시도하여 보강합니다.
    if not user_words:
        user_words = set(okt.nouns(question.lower()))

    if not user_words:
        return None
    
    best_score = 0.0
    best_answer = None
    
    for item in faq_data:
        faq_words = item.get("faq_words", set())
        
        if not faq_words:
            continue
        
        intersection = len(user_words.intersection(faq_words))
        union = len(user_words.union(faq_words))
        score = intersection / union if union > 0 else 0
        
        if score > best_score:
            best_score = score
            best_answer = item.get("answer")

    if best_score > 0.2:
        return best_answer
    return None

def fallback_pipeline(question: str) -> Dict[str, Any]:
    """키워드 매칭 및 FAQ 검색을 통해 간단한 질문에 답변하는 폴백 함수"""
    logger.info("fallback_pipeline_in", extra={"extra_data": {"q": question}})

    faq_answer = find_similar_faq(question)
    if faq_answer:
        prefix_message = PREFIX_MESSAGES["ok"]
        return {
            "result": prefix_message + faq_answer,
            "intent": "faq",
            "sources": [{"source": "faq_data.csv"}]
        }

    q = question.lower()
    if "비밀번호" in q or "초기화" in q:
        prefix_message = PREFIX_MESSAGES["ok"]
        intent = "reset_password"
        tool_output = tool_reset_password({})
    elif "id" in q or "계정" in q or "아이디" in q or "발급" in q:
        prefix_message = PREFIX_MESSAGES["ok"]
        intent = "request_id"
        tool_output = tool_request_id({})
    elif "담당자" in q:
        prefix_message = PREFIX_MESSAGES["ok"]
        screen = ""
        if "인사시스템" in q: screen = "인사시스템-사용자관리"
        elif "재무시스템" in q: screen = "재무시스템-정산화면"
        elif "포털" in q: screen = "포털-공지작성"
        
        intent = "owner_lookup"
        if screen:
            tool_output = tool_owner_lookup({"screen": screen})
            res = tool_output
            text = f"👤 '{res.get('screen')}' 담당자\n- 이름: {res.get('owner', {}).get('owner')}\n- 이메일: {res.get('owner', {}).get('email')}\n- 연락처: {res.get('owner', {}).get('phone')}" if res.get("ok") else f"❗{res.get('message','조회 실패')}"
        else: # 담당자 조회만 요청했을 경우
            all_owners_text = "✨ **담당자 조회 가능 목록** ✨\n\n"
            for s, info in OWNER_FALLBACK.items():
                all_owners_text += f"**- {s.split('-')[0]} 담당자:** {info.get('owner')}\n"
            all_owners_text += "\n\n**Tip:** '인사시스템 담당자 누구야?'처럼 구체적인 시스템명을 입력하면 더 자세한 정보를 얻을 수 있습니다."
            text = all_owners_text
            return {"result": prefix_message + text, "intent": intent, "sources": []}
    else:
        prefix_message = PREFIX_MESSAGES["fail"]
        no_match_message = "문의하신 내용에 대한 정보는 현재 답변이 어렵습니다.\n지원되는 기능과 관련된 내용으로 다시 질문해주시거나, 추가 문의는 고객센터를 이용해주세요."
        return {
            "result": prefix_message + no_match_message,
            "intent": "fallback_no_match",
            "sources": []
        }

    res = tool_output
    if intent == "reset_password":
        text = f"✅ 비밀번호 초기화 안내\n\n" + "\n".join(f"{i+1}. {s}" for i,s in enumerate(res.get("steps", []))) if res.get("ok") else f"❗{res.get('message','실패')}"
    elif intent == "request_id":
        text = f"🆔 ID 발급 신청\n\n" + "\n".join(f"{i+1}. {s}" for i,s in enumerate(res.get("steps", []))) if res.get("ok") else f"❗{res.get('message','실패')}"
    else:
        text = f"👤 '{res.get('screen')}' 담당자\n- 이름: {res.get('owner', {}).get('owner')}\n- 이메일: {res.get('owner', {}).get('email')}\n- 연락처: {res.get('owner', {}).get('phone')}" if res.get("ok") else f"❗{res.get('message','조회 실패')}"

    return {"result": prefix_message + text, "intent": intent, "sources": []}

_graph = None
def run_graph_pipeline(question: str, session_id: str) -> Dict[str, Any]:
    # [checklist: 5] LangChain & LangGraph - 멀티턴 대화 (memory) 활용
    """LangGraph 기반의 AI 파이프라인을 실행합니다."""
    global _graph
    logger.info("pipeline_in", extra={"extra_data": {"q": question}})
    if _graph is None: _graph = build_graph()
    out = _graph.invoke(
        input={"question": question, "intent":"", "result":"", "sources":[], "tool_output":{}},
        config={"configurable": {"thread_id": session_id}}
    )
    logger.info("pipeline_out", extra={"extra_data": {"intent": out.get("intent", "")}})
    return out

def pipeline(question: str, session_id: str) -> Dict[str, Any]:
    """
    Azure 연결 상태에 따라 적절한 파이프라인으로 요청을 라우팅합니다.
    """
    corrected_question = question
    
    # 간단한 인사말에 대한 응답 처리
    if corrected_question.lower().strip() in GREETINGS:
        return {
            "result": "네 반갑습니다. 문의사항을 말씀해 주시면 제가 도와드릴게요.",
            "intent": "greeting",
            "sources": []
        }

    if AZURE_AVAILABLE:
        return run_graph_pipeline(corrected_question, session_id)
    else:
        return fallback_pipeline(corrected_question)