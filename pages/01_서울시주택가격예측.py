# -*- coding: utf-8 -*-
"""
==========================================================================
  서울시 자치구별 아파트 가격 예측기
==========================================================================
서울 열린데이터광장 Open API(부동산 실거래가 정보)를 이용해서
자치구·건물면적·층·건축년도로 아파트 거래가격을 예측하는 스트림릿 앱입니다.

* 초보자를 위해 곳곳에 한국어 주석을 달아두었습니다.
* Streamlit Cloud에 올릴 때는 "Settings → Secrets"에 아래처럼 등록하세요.

    SEOUL_API_KEY = "발급받은_인증키"

==========================================================================
"""

# --------------------------------------------------------------------
# 1) 필요한 라이브러리 불러오기
# --------------------------------------------------------------------
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------------------------
# 2) 페이지 기본 설정
# --------------------------------------------------------------------
st.set_page_config(
    page_title="서울시 자치구별 아파트 가격 예측기",
    page_icon="🏢",
    layout="wide",
)

st.title("🏢 서울시 자치구별 아파트 가격 예측기")
st.caption("서울 열린데이터광장 실거래가 데이터를 학습해서, 자치구·면적·층·건축년도로 아파트 가격을 예측해요.")


# --------------------------------------------------------------------
# 3) 서울 열린데이터광장 Open API 기본 설정
# --------------------------------------------------------------------
BASE_URL = "http://openapi.seoul.go.kr:8088"
SERVICE = "tbLnOpendataRtmsV"
MAX_PER_CALL = 1000  # API 한 번 호출당 최대로 받을 수 있는 행(row) 개수
CURRENT_YEAR = datetime.now().year
EARLIEST_YEAR = 2006  # 실거래가 신고제도 시행연도(대략)

# 컬럼 이름 정리 (API 원래 이름 → 우리가 코드에서 쓸 이름)
# CGG_NM=자치구, STDG_NM=법정동, BLDG_NM=건물명, CTRT_DAY=계약일,
# THING_AMT=물건금액(만원), ARCH_AREA=건물면적(㎡), FLR=층,
# ARCH_YR=건축년도, BLDG_USG=건물용도, RTRCN_DAY=취소일

# 서울시 25개 자치구의 "공식 법정동코드(CGG_CD)"입니다.
# ⚠️ 인터넷에서 구할 수 있는 geojson 지도 파일 안의 code 값은 통계청 내부 코드라서
#    이 API가 쓰는 CGG_CD와 다를 수 있습니다. 그래서 아래처럼 API가 실제로 쓰는
#    공식 코드를 코드에 직접 넣어서 사용합니다 (강남구 11680, 은평구 11380,
#    구로구 11530, 강동구 11740 등 — API 응답 예시와 대조해서 검증된 값입니다).
OFFICIAL_GU_CODE = {
    "종로구": "11110", "중구": "11140", "용산구": "11170", "성동구": "11200",
    "광진구": "11215", "동대문구": "11230", "중랑구": "11260", "성북구": "11290",
    "강북구": "11305", "도봉구": "11320", "노원구": "11350", "은평구": "11380",
    "서대문구": "11410", "마포구": "11440", "양천구": "11470", "강서구": "11500",
    "구로구": "11530", "금천구": "11545", "영등포구": "11560", "동작구": "11590",
    "관악구": "11620", "서초구": "11650", "강남구": "11680", "송파구": "11710",
    "강동구": "11740",
}
CODE_TO_GU_NAME = {v: k for k, v in OFFICIAL_GU_CODE.items()}

COLUMN_TYPES = {
    "RCPT_YR": "str", "CGG_CD": "str", "CGG_NM": "str", "STDG_CD": "str", "STDG_NM": "str",
    "BLDG_NM": "str", "CTRT_DAY": "str", "THING_AMT": "float", "ARCH_AREA": "float",
    "FLR": "float", "RTRCN_DAY": "str", "ARCH_YR": "float", "BLDG_USG": "str",
}


# --------------------------------------------------------------------
# 4) API 호출 함수들 (페이지네이션 + 자치구별 재시도 포함)
# --------------------------------------------------------------------
def build_url(api_key, start, end, rcpt_yr="", cgg_cd=""):
    """요청 주소를 만듭니다.
    CGG_CD는 RCPT_YR 바로 다음 자리라서, 중간에 빈 칸이 생기지 않고
    안정적으로 자치구 필터가 걸립니다."""
    parts = [api_key, "json", SERVICE, str(start), str(end)]
    tail = [str(rcpt_yr), str(cgg_cd)]
    while tail and tail[-1] == "":
        tail.pop()
    parts += tail
    return BASE_URL + "/" + "/".join(parts)


def _parse_response(data):
    """API 응답에서 (행 목록, 전체 건수, 에러 메시지)를 뽑아냅니다."""
    if SERVICE not in data:
        for v in data.values():
            if isinstance(v, dict) and "RESULT" in v:
                r = v["RESULT"]
                return [], None, f"{r.get('CODE')}: {r.get('MESSAGE')}"
        return [], None, f"알 수 없는 응답 형식: {str(data)[:200]}"

    block = data[SERVICE]
    result = block.get("RESULT", {})
    code = result.get("CODE", "")
    if code and code != "INFO-000":
        return [], None, f"{code}: {result.get('MESSAGE')}"
    total = block.get("list_total_count")
    total = int(total) if total is not None else None
    rows = block.get("row", [])
    return rows, total, None


def fetch_year(api_key, year, cgg_cd, max_pages):
    """특정 연도 + 자치구코드 데이터를 페이지네이션으로 모두 가져옵니다."""
    rows_all = []
    start = 1
    page = 0
    total_count = None
    while True:
        end = start + MAX_PER_CALL - 1
        url = build_url(api_key, start, end, rcpt_yr=year, cgg_cd=cgg_cd)
        try:
            resp = requests.get(url, timeout=25)
            data = resp.json()
        except Exception as e:
            return rows_all, total_count, f"요청 실패({year}년 {cgg_cd}): {e}"

        rows, total_new, err = _parse_response(data)
        if err:
            if "ERROR-200" in err or "결과값이 없습니다" in err or "데이터가 없습니다" in err:
                break
            return rows_all, total_count, err

        if total_new is not None:
            total_count = total_new
        rows_all.extend(rows)
        page += 1
        if len(rows) < MAX_PER_CALL:
            break
        if total_count is not None and start + MAX_PER_CALL > total_count:
            break
        if page >= max_pages:
            break
        start += MAX_PER_CALL
        time.sleep(0.03)
    return rows_all, total_count, None


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_all(api_key, years, gu_cd_list, max_pages_per_combo):
    """연도 x 자치구코드 조합으로 데이터를 모아 반환합니다.
    (자치구별로 나눠서 조회 + 실패 시 한 번 재시도해서, 특정 구만 누락되는 것을 방지합니다.)"""
    all_rows = []
    combos = [(y, g) for y in years for g in (gu_cd_list if gu_cd_list else [""])]
    errors = []
    gu_last_error = {}
    prog = st.progress(0.0, text="데이터 수집 준비 중...")
    n = len(combos)
    for i, (y, g) in enumerate(combos):
        label = f"{y}년" + (f" · {CODE_TO_GU_NAME.get(g, g)}" if g else " · 서울 전체")
        prog.progress(i / max(n, 1), text=f"수집 중: {label} ({i+1}/{n})")

        rows, total, err = fetch_year(api_key, y, g, max_pages_per_combo)
        if err:
            time.sleep(1.0)  # 일시적 오류일 수 있으니 잠깐 쉬었다가 한 번 더 시도
            rows, total, err = fetch_year(api_key, y, g, max_pages_per_combo)

        if err:
            errors.append(f"{label}: {err}")
            if g:
                gu_last_error[g] = err
        all_rows.extend(rows)
        time.sleep(0.1)
    prog.progress(1.0, text="수집 완료")
    time.sleep(0.2)
    prog.empty()
    return all_rows, errors, gu_last_error


def rows_to_df(rows):
    """API가 준 리스트를 pandas DataFrame으로 바꾸고, 필요한 전처리를 합니다."""
    if not rows:
        return pd.DataFrame(columns=list(COLUMN_TYPES.keys()))
    df = pd.DataFrame(rows)
    for col, typ in COLUMN_TYPES.items():
        if col not in df.columns:
            df[col] = np.nan
        if typ == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["CTRT_DATE"] = pd.to_datetime(df["CTRT_DAY"], format="%Y%m%d", errors="coerce")
    df["PRICE_EOK"] = df["THING_AMT"] / 10000.0  # 만원 -> 억원 (화면 표시용)
    df["연식"] = CURRENT_YEAR - df["ARCH_YR"]      # 건축된 지 몇 년 됐는지
    # 취소된 거래(취소일이 채워진 행)는 분석에서 제외
    is_cancelled = df["RTRCN_DAY"].fillna("").astype(str).str.strip() != ""
    df = df[~is_cancelled].copy()
    return df


# --------------------------------------------------------------------
# 5) 인증키 확인 (Secrets에서만 로드, 화면에는 노출하지 않음)
# --------------------------------------------------------------------
api_key = st.secrets.get("SEOUL_API_KEY", "") if hasattr(st, "secrets") else ""

if not api_key:
    st.error(
        "🔑 서울 열린데이터광장 인증키가 설정되어 있지 않아요.\n\n"
        "앱을 배포한 곳의 **Settings → Secrets**에 아래처럼 추가해주세요.\n\n"
        "```toml\nSEOUL_API_KEY = \"발급받은_인증키\"\n```"
    )
    st.stop()


# --------------------------------------------------------------------
# 6) 사이드바 - 데이터 조회 조건
# --------------------------------------------------------------------
st.sidebar.title("🔎 조회 조건")

years = st.sidebar.slider(
    "계약연도 범위", min_value=EARLIEST_YEAR, max_value=CURRENT_YEAR,
    value=(CURRENT_YEAR - 2, CURRENT_YEAR),
    help="기간이 넓을수록 예측 모델 학습에는 좋지만, API 호출 수와 대기 시간이 늘어납니다.",
)

gu_selected = st.sidebar.multiselect(
    "자치구 필터 (미선택 시 서울 전체 25개 구)",
    options=sorted(OFFICIAL_GU_CODE.keys()), default=[],
)

max_pages = st.sidebar.number_input(
    "연도·자치구 조합당 최대 페이지 수 (1페이지=최대 1,000건)",
    min_value=1, max_value=200, value=3,
    help="값이 클수록 학습 데이터가 많아지지만 (연도 수 × 자치구 수 × 이 값)만큼 API 호출이 늘어납니다.",
)

if st.sidebar.button("🔄 새로고침 (캐시 지우고 다시 조회)", use_container_width=True):
    st.cache_data.clear()

if gu_selected:
    effective_gu_cd_list = [OFFICIAL_GU_CODE[nm] for nm in gu_selected]
else:
    effective_gu_cd_list = list(OFFICIAL_GU_CODE.values())

year_list = [str(y) for y in range(years[0], years[1] + 1)]
with st.spinner("서울 열린데이터광장에서 실거래가 데이터를 불러오는 중..."):
    rows, fetch_errors, gu_last_error = fetch_all(
        api_key, tuple(year_list), tuple(effective_gu_cd_list), max_pages
    )
raw_df = rows_to_df(rows)

if gu_last_error:
    failed_names = ", ".join(sorted(CODE_TO_GU_NAME.get(cd, cd) for cd in gu_last_error))
    st.warning(
        f"⚠️ 다음 자치구는 API 호출이 실패해서 데이터가 부족할 수 있어요: **{failed_names}** "
        "(인증키 일일 호출 한도 초과 또는 일시적 서버 오류일 수 있습니다. 잠시 후 새로고침 해보세요.)"
    )
if fetch_errors:
    with st.expander(f"오류 상세 보기 ({len(fetch_errors)}건)"):
        for e in fetch_errors:
            st.write("- ", e)

# 아파트만 사용 + 학습에 꼭 필요한 값이 없는 행은 제외
apt_df = raw_df[raw_df["BLDG_USG"] == "아파트"].copy()
apt_df = apt_df.dropna(subset=["THING_AMT", "ARCH_AREA", "FLR", "ARCH_YR", "CGG_NM"])
apt_df = apt_df[apt_df["THING_AMT"] > 0]

if len(apt_df) < 30:
    st.warning(
        f"학습에 쓸 수 있는 아파트 거래 데이터가 {len(apt_df)}건밖에 없어요. "
        "조회 조건(기간·자치구·페이지 수)을 넓혀서 다시 시도해주세요."
    )
    st.stop()

st.caption(f"✅ 아파트 거래 {len(apt_df):,}건을 불러와서 모델을 학습합니다.")


# --------------------------------------------------------------------
# 7) 선형회귀 모델 학습 (자치구 원-핫 인코딩 + 로그 변환)
# --------------------------------------------------------------------
FEATURE_NUM = ["ARCH_AREA", "FLR", "연식"]
FEATURE_CAT = ["CGG_NM"]

X = apt_df[FEATURE_NUM + FEATURE_CAT]
y_log = np.log(apt_df["THING_AMT"])  # 물건금액(만원)을 로그로 변환해서 학습

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# 자치구는 원-핫 인코딩, 숫자 특성은 그대로 사용하는 파이프라인
preprocessor = ColumnTransformer(
    transformers=[
        ("gu_onehot", OneHotEncoder(handle_unknown="ignore"), FEATURE_CAT),
        ("num", "passthrough", FEATURE_NUM),
    ]
)
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", LinearRegression()),
])
model.fit(X_train, y_train)

y_pred_log_test = model.predict(X_test)
r2 = r2_score(y_test, y_pred_log_test)


def predict_price_eok(cgg_nm, arch_area, flr, age):
    """모델로 물건금액(억원)을 예측해서 돌려줍니다."""
    row = pd.DataFrame([{"ARCH_AREA": arch_area, "FLR": flr, "연식": age, "CGG_NM": cgg_nm}])
    pred_log = model.predict(row[FEATURE_NUM + FEATURE_CAT])[0]
    pred_manwon = np.exp(pred_log)  # 로그를 다시 원래 크기(만원)로 되돌리기
    return pred_manwon / 10000.0  # 억원으로 환산


# --------------------------------------------------------------------
# 8) R² 카드
# --------------------------------------------------------------------
st.markdown("---")
r2_col, exp_col = st.columns([1, 3])
with r2_col:
    st.metric("R² (결정계수)", f"{r2:.3f}")
with exp_col:
    st.info(
        "R²(결정계수)는 모델이 실제 가격 변화를 얼마나 잘 설명하는지 나타내는 값이에요. "
        "1에 가까울수록 예측이 정확하고, 0에 가까울수록 설명력이 부족하다는 뜻이에요."
    )


# --------------------------------------------------------------------
# 9) 예측 UI - 두 가지 방식 중 선택
# --------------------------------------------------------------------
st.markdown("---")
st.subheader("💰 아파트 가격 예측해보기")

mode = st.radio(
    "예측 방법을 선택하세요",
    ["📋 실거래 목록에서 선택", "✍️ 직접 입력"],
    horizontal=True,
)

if mode == "📋 실거래 목록에서 선택":
    # 자치구 -> 법정동 -> 개별 거래 순서로 좁혀가며 고르게 해서 목록이 너무 길어지지 않게 함
    col1, col2 = st.columns(2)
    with col1:
        pick_gu = st.selectbox("자치구", sorted(apt_df["CGG_NM"].unique()))
    dong_options = sorted(apt_df.loc[apt_df["CGG_NM"] == pick_gu, "STDG_NM"].dropna().unique())
    with col2:
        pick_dong = st.selectbox("법정동", dong_options)

    candidates = apt_df[(apt_df["CGG_NM"] == pick_gu) & (apt_df["STDG_NM"] == pick_dong)].copy()
    candidates["표시이름"] = (
        candidates["BLDG_NM"].fillna("(건물명 미상)") + " · "
        + candidates["CTRT_DATE"].dt.strftime("%Y-%m-%d").fillna("") + " · "
        + candidates["PRICE_EOK"].round(2).astype(str) + "억원"
    )
    pick_label = st.selectbox("실제 거래 선택 (입력해서 검색할 수 있어요)", candidates["표시이름"])
    picked = candidates[candidates["표시이름"] == pick_label].iloc[0]

    pred_eok = predict_price_eok(picked["CGG_NM"], picked["ARCH_AREA"], picked["FLR"], picked["연식"])
    actual_eok = picked["PRICE_EOK"]

    st.write(
        f"**{picked['CGG_NM']} {picked['STDG_NM']} {picked['BLDG_NM']}** "
        f"(면적 {picked['ARCH_AREA']:.1f}㎡, {int(picked['FLR'])}층, 건축년도 {int(picked['ARCH_YR'])}년)"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("실제 거래가격", f"{actual_eok:.2f} 억원")
    with c2:
        st.metric("모델 예측가격", f"{pred_eok:.2f} 억원",
                   delta=f"{pred_eok - actual_eok:+.2f} 억원 (예측-실제)")

    st.markdown(
        f"<div style='text-align:center; margin-top:1.2em;'>"
        f"<span style='font-size:1.1em;'>🔮 예측 가격</span><br>"
        f"<span style='font-size:3em; font-weight:800; color:#e0522f;'>{pred_eok:.2f}억원</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

else:
    pick_gu = st.selectbox("자치구", sorted(OFFICIAL_GU_CODE.keys()), key="direct_gu")

    c1, c2, c3 = st.columns(3)
    with c1:
        area = st.slider("건물면적(㎡)", min_value=20.0, max_value=250.0, value=84.0, step=1.0)
    with c2:
        floor = st.slider("층", min_value=-3, max_value=50, value=10, step=1)
    with c3:
        age = st.slider("연식(건축 후 경과년수)", min_value=0, max_value=60, value=15, step=1)

    pred_eok = predict_price_eok(pick_gu, area, floor, age)

    st.markdown(
        f"<div style='text-align:center; margin-top:1.2em;'>"
        f"<span style='font-size:1.1em;'>🔮 {pick_gu} · {area:.0f}㎡ · {floor}층 · 연식 {age}년 예측 가격</span><br>"
        f"<span style='font-size:3.2em; font-weight:800; color:#e0522f;'>{pred_eok:.2f}억원</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "본 모델은 자치구·건물면적·층·건축년도(연식) 네 가지만 사용한 단순 선형회귀이며, "
    "실제 시세와는 차이가 있을 수 있습니다. 물건금액은 로그 변환 후 학습했고, "
    "예측값은 다시 지수변환(exp)해서 원래 크기(억원)로 보여드립니다."
)
