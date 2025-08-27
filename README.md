# 💼 사내 헬프데스크 챗봇 (RAG + LangGraph)

이 프로젝트는 LangChain, FastAPI, Streamlit을 활용하여 구축된 사내 헬프데스크 챗봇 애플리케이션입니다. RAG(검색 증강 생성) 기술을 사용하여 사내 문서 기반의 답변을 제공하고, 특정 요청에 대해서는 미리 정의된 기능을 실행하는 에이전트 역할을 수행합니다.

-   **UI**: Streamlit
-   **Backend API**: FastAPI
-   **오케스트레이션**: LangGraph (인텐트 라우팅)
-   **검색**: FAISS + LangChain (PDF/CSV/TXT/DOCX 지원)
-   **모델**: Azure OpenAI (AOAI)

---

## ✨ 주요 기능

-   **대화형 서비스**: Streamlit UI를 통해 사용자 친화적인 챗봇 대화 환경을 제공합니다.
-   **RAG (Retrieval-Augmented Generation)**: `./kb` 폴더의 사내 매뉴얼이나 문서를 기반으로 정확하고 관련성 높은 답변을 생성합니다.
-   **LangGraph 기반 인텐트 라우팅**: 복잡한 사용자 의도를 분석하고, 일반 질의응답과 특정 기능 실행(툴 사용)을 유연하게 처리합니다.
-   **미리 정의된 기능 (Tools)**:
    -   ID 발급 신청 안내
    -   비밀번호 초기화 절차 안내
    -   업무/화면별 담당자 정보 조회

---

## 📄 프로젝트 구조

전문적인 개발 및 유지보수를 위해 소스코드(`src`), Docker 설정(`docker`), 테스트(`tests`)를 명확히 분리한 구조를 따릅니다.

```
helpdeskbot/
├── .dockerignore
├── .env
├── docker-compose.yml
├── README.md
├── pyproject.toml         # 📦 프로젝트 의존성 및 메타데이터 관리
├── docker/                # 🐳 Dockerfile 관리
│   ├── Dockerfile.api
│   └── Dockerfile.ui
├── kb/                    # 📚 RAG 학습을 위한 원본 문서 폴더
├── index/                 # 🗂️ FAISS 벡터 인덱스 저장 폴더
├── src/                   # 🐍 파이썬 소스코드
│   └── helpdeskbot/      #   └── 파이썬 패키지
│       ├── __init__.py
│       ├── api.py         # FastAPI 백엔드
│       ├── core.py        # RAG, LangGraph 핵심 로직
│       └── ui.py          # Streamlit 프론트엔드
└── tests/                 # 🧪 테스트 코드
    ├── __init__.py
    └── test_api.py
```

---

## 🛠️ 설치 및 실행

### 1. 가상환경 및 의존성 설치

`pyproject.toml`을 사용하여 프로젝트 의존성을 관리합니다.

```bash
# 1. 가상환경 생성 및 활성화
#python -m venv .venv
#source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. 의존성 설치 (운영용)
pip install .

# 3. 의존성 설치 (개발용 - 테스트 라이브러리 포함)
pip install -e ".[test]"
```

### 2. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 자신의 Azure OpenAI 정보로 채워주세요.

```bash
# .env

# Azure OpenAI Environment Variables
AOAI_ENDPOINT=[https://your-aoai-endpoint.openai.azure.com/](https://your-aoai-endpoint.openai.azure.com/)
AOAI_API_KEY=your-aoai-api-key
AOAI_API_VERSION=2024-10-21

# Deployments
AOAI_DEPLOY_GPT4O_MINI=gpt-4o-mini
AOAI_DEPLOY_GPT4O=gpt-4o
AOAI_DEPLOY_EMBED_3_SMALL=text-embedding-3-small

# API Server Configuration
API_SERVER_HOST=0.0.0.0
API_CLIENT_HOST=localhost
API_PORT=8001
```

### 3. 애플리케이션 실행

각각 별도의 터미널에서 실행해야 합니다.

**1) FastAPI 백엔드 실행**
```bash
python -m helpdesk_bot.api --port 8001 &
```

**2) Streamlit UI 실행**
```bash
streamlit run src/helpdesk_bot/ui.py --server.port 8507
```

-   UI 접속: [http://localhost:8507](http://localhost:8507)

---

## 🧪 테스트

프로젝트의 안정성을 보장하기 위해 `pytest`를 사용합니다.

```bash
# tests 폴더의 모든 테스트 실행
pytest
```

---

## 🛠️ 로컬 환경 실행 및 테스트 가이드

Docker를 사용하지 않고 로컬 환경에서 프로젝트를 설정하고 테스트하는 전체 과정입니다.

### 1단계: 프로젝트 초기 설정
프로젝트를 위한 격리된 파이썬 가상환경을 생성하고 활성화합니다.

```bash
# 1. 가상환경 생성
python -m venv .venv

# 2. 가상환경 활성화
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# 가상화 종료
deactivate
```

### 2단계: 의존성 설치
`pyproject.toml`을 사용하여 프로젝트 실행과 테스트에 필요한 모든 라이브러리를 설치합니다.

| 구분 | `pip install .` | `pip install -e ".[test]"` |
| :--- | :--- | :--- |
| **목적** | **운영/실행** | **개발/테스트** |
| **설치 대상** | 필수 라이브러리만 | 필수 + **테스트** 라이브러리 |
| **코드 수정** | 재설치 필요 | **즉시 반영 (Editable)** |
| **사용 시점** | 서버 배포, Docker 이미지 빌드 | 내 PC에서 코딩 및 테스트 |


```bash
# 1. 의존성 설치 (운영용)
pip install .

# 2. 의존성 설치 (개발용 - 테스트 라이브러리 포함)
pip install -e .
pip install -e ".[test]"
```

### 3단계: 환경 변수 설정
프로젝트 최상위 폴더에 `.env` 파일을 생성하고, 자신의 Azure OpenAI 정보로 값을 수정해야 합니다.

```bash
# .env

# Azure OpenAI Environment Variables
AOAI_ENDPOINT=[https://your-aoai-endpoint.openai.azure.com/](https://your-aoai-endpoint.openai.azure.com/)
AOAI_API_KEY=your-aoai-api-key
AOAI_API_VERSION=2024-10-21

# Deployments
AOAI_DEPLOY_GPT4O_MINI=gpt-4o-mini
AOAI_DEPLOY_GPT4O=gpt-4o
AOAI_DEPLOY_EMBED_3_SMALL=text-embedding-3-small

# API Server Configuration
API_SERVER_HOST=0.0.0.0
API_CLIENT_HOST=localhost
API_PORT=8001
```

### 4단계: 애플리케이션 실행
API 서버와 UI를 각각 다른 터미널에서 실행해야 합니다. (각 터미널에서 가상환경 활성화 필요)

**- 터미널 1: FastAPI 백엔드 실행**
```bash
python -m helpdesk_bot.api --port 8001
```

**- 터미널 2: Streamlit UI 실행**
```bash
streamlit run src/helpdesk_bot/ui.py --server.port 8507
```
이제 웹 브라우저에서 **[http://localhost:8507](http://localhost:8507)** 주소로 접속하면 챗봇 UI를 사용할 수 있습니다.

### 5단계: 단위 테스트 실행
프로젝트 최상위 폴더에서 아래 명령어를 실행하여 코드의 안정성을 검증합니다.

```bash
pytest
```
`pytest`가 `tests` 폴더를 자동으로 찾아 모든 테스트를 실행하고, 전부 `PASSED`로 표시되면 성공입니다.


---
## 🐳 Docker 실행 가이드
`docker-compose`를 사용하여 API와 UI 컨테이너를 한 번에 실행할 수 있습니다.

### Dockerfile 예시
**`docker/Dockerfile.api`**
```dockerfile
FROM python:3.11-slim
WORKDIR /app

# pyproject.toml을 사용한 의존성 설치
COPY pyproject.toml /app/
RUN pip install --no-cache-dir .

# 소스코드 복사
COPY src/helpdesk_bot /app/helpdesk_bot

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "helpdesk_bot.api", "--host", "0.0.0.0", "--port", "8000"]
```

**`docker/Dockerfile.ui`**
```dockerfile
FROM python:3.11-slim
WORKDIR /app

# pyproject.toml을 사용한 의존성 설치
COPY pyproject.toml /app/
RUN pip install --no-cache-dir .

# 소스코드 복사
COPY src/helpdesk_bot /app/helpdesk_bot

ENV PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["streamlit", "run", "helpdesk_bot/ui.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

### docker-compose 실행
`.env` 파일이 올바르게 설정되었는지 확인 후, 아래 명령어를 실행하세요.

```bash
docker compose up --build
```

-   **UI 접속**: [http://localhost:8501](http://localhost:8501)
-   **API 문서 (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
-   `./kb`, `./index`, `./logs` 폴더는 **볼륨**으로 마운트되어 컨테이너 재시작 시에도 데이터가 유지됩니다.
