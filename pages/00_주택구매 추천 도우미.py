# -*- coding: utf-8 -*-
"""
==========================================================================
  Solar API 기반 AI 채팅 앱 (주택 구매 자치구·동 추천 도우미)
==========================================================================
이 앱은 업스테이지(Upstage)의 Solar API(solar-open2 모델)를 사용해서
챗봇처럼 대화하는 스트림릿(Streamlit) 앱입니다.

* 초보자를 위해 코드 곳곳에 한국어 주석을 달아두었습니다.
* Streamlit Cloud에 올릴 때는 API 키를 코드에 직접 적지 않고,
  "Settings → Secrets"에 아래처럼 등록해서 사용합니다.

    SOLAR_API_KEY = "여기에_발급받은_키를_붙여넣기"

==========================================================================
"""

# --------------------------------------------------------------------
# 1) 필요한 라이브러리 불러오기
# --------------------------------------------------------------------
import streamlit as st          # 화면(웹 UI)을 그려주는 라이브러리
from openai import OpenAI        # Solar API도 OpenAI와 같은 방식으로 호출 가능


# --------------------------------------------------------------------
# 2) 페이지 기본 설정
# --------------------------------------------------------------------
st.set_page_config(
    page_title="주택 구매 동네 추천 챗봇",
    page_icon="🏡",
    layout="centered",
)

st.title("🏡 주택 구매 동네 추천 챗봇")
st.caption("서울시 자치구·법정동 추천을 도와주는 따뜻한 데이터 분석 선생님이에요 :)")


# --------------------------------------------------------------------
# 3) Solar API 설정값
# --------------------------------------------------------------------
# - MODEL_NAME은 절대 다른 이름으로 바꾸지 말라고 하셨으니 그대로 사용합니다.
# - BASE_URL은 Upstage Solar API 주소입니다.
MODEL_NAME = "solar-open2"
BASE_URL = "https://api.upstage.ai/v1"

# 챗봇의 성격(시스템 프롬프트)을 정해줍니다.
# 요청하신 문구를 그대로 넣고, 이 앱의 핵심 목적인
# "주택 구매 자치구·동 추천"에 도움이 되도록 안내를 살짝 덧붙였습니다.
SYSTEM_PROMPT = (
    "너는 따뜻하고 친절한 데이터 분석 선생님이야. 반드시 순수 한국어로만 답해. "
    "사용자가 주택을 구매하려는 자치구나 법정동 추천을 원하면, "
    "예산, 직장/통근 위치, 교통, 학군, 편의시설, 선호하는 분위기 등을 "
    "친근하게 물어보면서 근거와 함께 차근차근 추천해줘."
)


# --------------------------------------------------------------------
# 4) 비밀 금고(secrets)에서 API 키 가져오기
# --------------------------------------------------------------------
# 코드에 키를 직접 쓰지 않고, Streamlit의 st.secrets를 통해서만 불러옵니다.
# Streamlit Cloud에서는 "Settings → Secrets"에 SOLAR_API_KEY를 등록하면 됩니다.
api_key = st.secrets.get("SOLAR_API_KEY", "")

if not api_key:
    st.error(
        "🔑 Solar API 키가 설정되어 있지 않아요.\n\n"
        "앱을 배포한 곳의 **Settings → Secrets**에 아래처럼 추가해주세요.\n\n"
        "```toml\nSOLAR_API_KEY = \"발급받은_API_키\"\n```"
    )
    st.stop()  # 키가 없으면 여기서 앱 실행을 멈춥니다.

# openai 라이브러리의 클라이언트를 만들되, 접속 주소만 Solar API로 바꿔줍니다.
client = OpenAI(api_key=api_key, base_url=BASE_URL)


# --------------------------------------------------------------------
# 5) 대화 기록을 세션(session_state)에 저장하기
# --------------------------------------------------------------------
# st.session_state는 사용자가 새 메시지를 보내도 브라우저 새로고침 전까지
# 값이 계속 유지되는 "저장 공간"입니다. 여기에 대화 기록을 쌓아둡니다.
if "messages" not in st.session_state:
    # 처음 실행할 때는 시스템 프롬프트만 들어있는 상태로 시작합니다.
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]


# --------------------------------------------------------------------
# 6) 지금까지의 대화 내용을 화면에 말풍선으로 그려주기
# --------------------------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue  # 시스템 프롬프트는 화면에 보여주지 않습니다.
    with st.chat_message(msg["role"]):  # role은 "user" 또는 "assistant"
        st.markdown(msg["content"])


# --------------------------------------------------------------------
# 7) 채팅 입력창 + AI 응답 처리
# --------------------------------------------------------------------
user_input = st.chat_input("궁금한 동네나 조건을 입력해보세요! (예: 예산 5억, 강남 출퇴근 30분 이내)")

if user_input:
    # 7-1) 사용자가 보낸 메시지를 대화 기록에 추가하고, 말풍선으로 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 7-2) AI의 답변을 스트리밍(실시간 타이핑)으로 보여줄 자리
    with st.chat_message("assistant"):
        placeholder = st.empty()   # 답변 글자가 채워질 빈 자리
        full_answer = ""            # 지금까지 도착한 글자를 계속 이어붙일 변수

        try:
            # Solar API에 대화 기록 전체를 보내서 답변을 요청합니다.
            # stream=True 로 설정하면 답이 한 번에 오지 않고 조금씩 나눠서 옵니다.
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=st.session_state.messages,
                stream=True,
                # temperature가 아니라 reasoning_effort로 "생각(추론) 기능"을 끕니다.
                # 'none'으로 주면 모델이 깊게 고민하지 않고 빠르게 답을 내놓습니다.
                extra_body={"reasoning_effort": "none"},
            )

            # 스트림에서 조각(chunk)이 도착할 때마다 화면을 갱신합니다.
            for chunk in stream:
                # 혹시 내용이 비어있는 조각이 올 수도 있으니 안전하게 확인합니다.
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    full_answer += piece
                    # 타이핑 되는 느낌을 주기 위해 커서(▌)를 끝에 붙여줍니다.
                    placeholder.markdown(full_answer + "▌")

            # 스트리밍이 끝나면 커서를 떼고 최종 답변만 보여줍니다.
            placeholder.markdown(full_answer)

        except Exception:
            # API 호출이 실패했을 때, 에러 메시지를 그대로 보여주지 않고
            # 사용자가 놀라지 않도록 친절한 한국어 안내문을 보여줍니다.
            full_answer = (
                "죄송해요, 지금은 답변을 가져오는 데 문제가 생겼어요. 😥\n\n"
                "인터넷 연결이나 API 키 설정을 확인한 뒤, 잠시 후 다시 시도해주세요."
            )
            placeholder.markdown(full_answer)

    # 7-3) AI의 답변도 대화 기록에 저장해서, 다음 대화에서 이어갈 수 있게 합니다.
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
