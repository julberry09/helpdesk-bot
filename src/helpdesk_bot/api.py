# api.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))))

import uvicorn
import time as _time
from typing import List, Dict, Any

from fastapi import FastAPI, Body, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# 공통 로직(파이프라인, 로거) 임포트
from src.helpdesk_bot.core import pipeline, logger

# =============================================================
# 1. FastAPI 앱 설정
# =============================================================
# [checklist: 11] 서비스 개발 및 패키징 - FastAPI를 활용하여 백엔드 API 구성
api = FastAPI(title="Helpdesk RAG API", version="0.1.0")

class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = _time.time()
        logger.info("api_request", extra={"extra_data": {"path": request.url.path}})
        try:
            response = await call_next(request)
            dur = round((_time.time() - start)*1000)
            logger.info("api_response", extra={"extra_data": {"status": response.status_code, "ms": dur}})
            return response
        except Exception as e:
            logger.exception("api_error", extra={"extra_data": {"error": str(e)}})
            raise

api.add_middleware(AuditMiddleware)

# =============================================================
# 2. API 엔드포인트
# =============================================================
# [checklist: 5] LangChain & LangGraph - 멀티턴 대화 (memory) 활용
# 💡 수정: ChatIn 모델에 session_id 필드 추가
class ChatIn(BaseModel): message: str; session_id: str

class ChatOut(BaseModel): reply: str; intent: str; sources: List[Dict[str, Any]]= []

@api.get("/health")
def health(): return {"ok":True}

@api.post("/chat", response_model=ChatOut)
# 💡 수정: chat 함수에서 payload의 session_id를 추출하여 pipeline에 전달
def chat(payload: ChatIn = Body(...)):
    out = pipeline(payload.message, payload.session_id)
    return ChatOut(reply=out.get("result",""), intent=out.get("intent",""), sources=out.get("sources", []))

# =============================================================
# 3. 엔트리포인트
# =============================================================
if __name__ == "__main__":
    import argparse
    default_host = os.getenv("API_SERVER_HOST", "0.0.0.0")
    default_port = int(os.getenv("API_PORT", 8000))

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", default=default_port, type=int)
    args = parser.parse_args()

    uvicorn.run(api, host=args.host, port=args.port)