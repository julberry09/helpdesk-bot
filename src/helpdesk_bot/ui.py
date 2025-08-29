# src/helpdesk_bot/ui.py

# 환경 경로 문제 해결을 위한 코드
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))))

import streamlit as st
import httpx

# 절대 경로 임포트 사용
from helpdesk_bot.core import pipeline, KB_DATA_DIR, INDEX_DIR, INDEX_NAME, build_or_load_vectorstore, AZURE_AVAILABLE

# src/helpdesk_bot/ui.py

def format_source_name(source_name: str) -> str:
    """
    파일 이름을 사용자가 이해하기 쉬운 설명으로 변환합니다.
    """
    # 미리 정의된 파일 이름에 대한 설명
    known_sources = {
        "faq_data.csv": "자주 묻는 질문(FAQ)"
       #, "seed-faq.txt": "표준 답변 자료"
    }
    
    # 알려진 소스이면 정의된 설명을 사용, 아니면 '참고 문서'라는 일반적인 용어 사용
    display_name = known_sources.get(source_name, "참고 문서")
    
    return f"{display_name} (파일명: {source_name})"

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
    # 버튼 텍스트를 왼쪽 정렬하는 CSS
    st.markdown("""
    <style>
        /* stButton 클래스 바로 아래 button 요소의 정렬 방식을 강제로 왼쪽으로 변경 */
        .stButton>button {
            justify-content: flex-start !important;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("🎓 챗봇 학습시키기")

        # 전체적인 들여쓰기를 제어할 컬럼 생성
        left_space, main_col = st.columns([0.01, 0.99])

        # 파일 업로더와 버튼을 모두 오른쪽 컬럼(main_col) 안에 배치하여 시작점 통일
        with main_col:
            uploaded = st.file_uploader(
                "문서 업로드 (PDF/CSV/TXT/DOCX)", 
                type=["pdf","csv","txt","md","docx"], 
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            if uploaded:
                KB_DATA_DIR.mkdir(parents=True, exist_ok=True)
                for f in uploaded:
                    with open(KB_DATA_DIR / f.name, "wb") as w:
                        w.write(f.read())
                st.success(f"{len(uploaded)}개 문서 저장됨. 'Rebuild Index'를 눌러 반영하세요.")

            # 버튼 좌우에 5% 여백을 주기 위한 중첩 컬럼
            btn_left, btn_mid, btn_right = st.columns([0.05, 0.7, 0.25])
            # '인덱스 재빌드 Rebuild Index' 버튼을 가운데 컬럼(btn_mid)에 배치
            with btn_mid:
                if st.button("Sync Content", disabled=not AZURE_AVAILABLE, use_container_width=True):
                    try:
                        for ext in [".faiss", ".pkl"]:
                            p = INDEX_DIR / f"{INDEX_NAME}{ext}"
                            if p.exists(): p.unlink()
                        with st.spinner("Index 재생성 중..."):
                            build_or_load_vectorstore()
                        st.success("완료!")
                    except Exception as e:
                        st.error(f"실패: {e}")
        
        if not AZURE_AVAILABLE:
            st.caption("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Sync Content 는 AI 분석 서버")
            st.caption("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;연결 시에만 활성화됩니다.")

        st.divider()

        api_host = os.getenv("API_CLIENT_HOST", "localhost")
        api_port = int(os.getenv("API_PORT", 8001))
        api_base_url = f"http://{api_host}:{api_port}"
        
        # 1. st.status는 "확인 중..." 메시지를 잠깐 보여주는 용도로만 사용
        with st.status("시스템 상태 확인 중...", expanded=False) as status:
            api_is_healthy = check_api_health(api_base_url)
            # 확인이 끝나면 이 메시지는 자동으로 사라집니다.
            status.update(label="확인 완료", state="complete")

            # 2. 확인된 결과를 바탕으로 st.markdown을 사용해 직접 아이콘과 텍스트 표시
            if api_is_healthy and AZURE_AVAILABLE:
                st.markdown("✅ AI 서버: 온라인")
            elif api_is_healthy and not AZURE_AVAILABLE:
                st.markdown("⚠️ AI 서버: 제한 모드 (기본 답변만 가능)")
            else:
                st.markdown("🚨 API 서버: 오프라인")

    for role, content in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(content)

    if q := st.chat_input("궁금한 것을 입력해주세요..."):
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"): st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("처리 중..."):
                # [수정 시작] ----------------------------------------------------------------
                reply, sources = None, []
                try:
                    # line 125: api_is_healthy가 True이면 API 호출 시도
                    if api_is_healthy:
                        with httpx.Client(timeout=30.0) as client:
                            resp = client.post(f"{api_base_url}/chat", json={"message": q})
                            resp.raise_for_status()
                            data = resp.json()
                            reply = data.get("reply",""); sources = data.get("sources", [])
                    else:
                        out = pipeline(q)
                        reply = out.get("result",""); sources = out.get("sources", [])

                # line 136: API 연결 실패 에러를 별도로 잡아서 처리
                except httpx.ConnectError:
                    st.warning("API 서버에 연결할 수 없어 로컬 폴백 모드로 자동 전환하여 재시도합니다.")
                    # API 호출 실패 시, 로컬 파이프라인으로 다시 시도
                    out = pipeline(q)
                    reply = out.get("result",""); sources = out.get("sources", [])
                
                # line 143: 그 외 예상치 못한 다른 에러 처리
                except Exception as e:
                    reply = f"오류: {e}"
                    sources = []

                # line 147: 응답 및 소스 표시 로직
                if reply:
                    st.markdown(reply)
                    if sources:
                        with st.expander("🔎 참고 자료"):
                            for s in sources:
                                source_display = format_source_name(s.get('source', '알 수 없음'))
                                if s.get("page") is not None:
                                    line = f"- {source_display}, page {int(s['page']) + 1}"
                                else:
                                    line = f"- {source_display}"
                                st.write(line)
                    st.session_state.chat.append(("assistant", reply))
                # [수정 끝] ------------------------------------------------------------------


if __name__ == "__main__":
    main()