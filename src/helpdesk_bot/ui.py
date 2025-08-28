# src/helpdesk_bot/ui.py

# 환경 경로 문제 해결을 위한 코드
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))))

import streamlit as st
import httpx

# 절대 경로 임포트 사용
from helpdesk_bot.core import pipeline, KB_DATA_DIR, INDEX_DIR, INDEX_NAME, build_or_load_vectorstore, AZURE_AVAILABLE

# API 상태를 확인하는 함수 (60초 동안 결과를 캐시하여 성능 저하 방지)
@st.cache_data(ttl=60)
def check_api_health(api_base_url):
    """API 서버의 /health 엔드포인트를 확인하여 상태를 반환합니다."""
    try:
        with httpx.Client(timeout=2) as client:
            resp = client.get(f"{api_base_url}/health")
            return resp.status_code == 200
    except httpx.ConnectError:
        return False

def main():
    st.set_page_config(page_title="사내 헬프데스크 챗봇", page_icon="💡", layout="wide")
    st.title("🌞 사내 헬프데스크 챗봇")

    with st.sidebar:
        st.header("📚 지식베이스(KB)")

        # [수정 시작] ---------------------------------------------

        # 1. 전체적인 들여쓰기를 제어할 컬럼을 생성합니다.
        #    첫 번째 컬럼은 비워두어 왼쪽 여백(들여쓰기) 역할을 합니다.
        left_space, main_col = st.columns([0.1, 0.9])

        # 2. 파일 업로더와 버튼을 모두 오른쪽 컬럼(main_col) 안에 배치합니다.
        with main_col:
            uploaded = st.file_uploader(
                "문서 업로드 (PDF/CSV/TXT/DOCX)", 
                type=["pdf","csv","txt","md","docx"], 
                accept_multiple_files=True,
                label_visibility="collapsed" # ◀◀◀ 라벨을 숨겨 UI를 더 깔끔하게 만듭니다.
            )
            if uploaded:
                KB_DATA_DIR.mkdir(parents=True, exist_ok=True)
                for f in uploaded:
                    with open(KB_DATA_DIR / f.name, "wb") as w:
                        w.write(f.read())
                st.success(f"{len(uploaded)}개 문서 저장됨. '인덱스 재빌드'를 눌러 반영하세요.")

            # 버튼도 같은 컬럼 안에 배치하고, use_container_width를 사용해 너비를 맞춥니다.
            if st.button("인덱스 재빌드", disabled=not AZURE_AVAILABLE, use_container_width=True):
                try:
                    # ... (인덱스 재빌드 로직)
                except Exception as e:
                    st.error(f"실패: {e}")
        
        # [수정 끝] -----------------------------------------------
        
        if not AZURE_AVAILABLE:
            st.caption("ℹ️ '인덱스 재빌드'는 Azure 연결 시에만 활성화됩니다.")

        st.divider()
        api_host = os.getenv("API_CLIENT_HOST", "localhost")
        api_port = int(os.getenv("API_PORT", 8001))
        api_base_url = f"http://{api_host}:{api_port}"
        
        api_is_healthy = check_api_health(api_base_url)

        if api_is_healthy:
            st.status("API 서버: 온라인", state="complete")
        else:
            st.status("API 서버: 오프라인 (폴백 모드)", state="error")


    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, content in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(content)

    if q := st.chat_input("무엇을 도와드릴까요?"):
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"): st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("처리 중..."):
                try:
                    if api_is_healthy:
                        with httpx.Client(timeout=30.0) as client:
                            resp = client.post(f"{api_base_url}/chat", json={"message": q})
                            resp.raise_for_status()
                            data = resp.json()
                            reply = data.get("reply",""); sources = data.get("sources", [])
                    else:
                        out = pipeline(q)
                        reply = out.get("result",""); sources = out.get("sources", [])
                    
                    st.markdown(reply)
                    if sources:
                        with st.expander("🔎 참조 소스"):
                            for s in sources:
                                line = f"- [{s.get('index')}] {s.get('source')}"
                                if s.get("page") is not None: line += f" (page {s['page']})"
                                st.write(line)
                    st.session_state.chat.append(("assistant", reply))
                except Exception as e:
                    st.error(f"오류: {e}")

if __name__ == "__main__":
    main()