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
import plotly.express as px
import plotly.graph_objects as go
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

# 테두리가 있는 st.container(border=True)를 하얀 배경 + 둥근 모서리 + 은은한 그림자 카드처럼 보이게 꾸며줍니다.
st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        padding: 0.5em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def postprocess_df(df):
    """API 원래 컬럼 이름(CGG_NM 등)을 가진 DataFrame에 공통 전처리를 적용합니다.
    (API로 막 받아온 데이터든, CSV에서 불러온 데이터든 이 함수를 거치면 똑같은 형태가 됩니다)"""
    for col, typ in COLUMN_TYPES.items():
        if col not in df.columns:
            df[col] = np.nan
        if typ == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["CTRT_DATE"] = pd.to_datetime(df["CTRT_DAY"], format="%Y%m%d", errors="coerce")
    df["CTRT_YEAR"] = df["CTRT_DATE"].dt.year  # 계약연도 (실제 가격 추이 그래프용)
    df["PRICE_EOK"] = df["THING_AMT"] / 10000.0  # 만원 -> 억원 (화면 표시용)
    df["연식"] = CURRENT_YEAR - df["ARCH_YR"]      # 건축된 지 몇 년 됐는지
    # 취소된 거래(취소일이 채워진 행)는 분석에서 제외
    is_cancelled = df["RTRCN_DAY"].fillna("").astype(str).str.strip() != ""
    df = df[~is_cancelled].copy()
    return df


def rows_to_df(rows):
    """API가 준 리스트(list of dict)를 DataFrame으로 바꿉니다."""
    if not rows:
        return pd.DataFrame(columns=list(COLUMN_TYPES.keys()))
    return postprocess_df(pd.DataFrame(rows))


# CSV 파일의 한글 컬럼명 -> API 원래 컬럼명으로 되돌리는 매핑
# (colab_fetch_seoul_csv.py 로 만든 seoul.csv 파일과 짝을 이룹니다)
CSV_COLUMN_TO_API = {
    "자치구": "CGG_NM", "법정동": "STDG_NM", "건물명": "BLDG_NM", "계약일": "CTRT_DAY",
    "물건금액": "THING_AMT", "건물면적": "ARCH_AREA", "층": "FLR", "건축년도": "ARCH_YR",
    "건물용도": "BLDG_USG", "취소일": "RTRCN_DAY",
}


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def csv_to_df(csv_url):
    """미리 만들어둔 CSV(예: colab_fetch_seoul_csv.py로 생성)를 읽어옵니다. API 호출 없이 빠릅니다.
    주소가 .gz로 끝나면 pandas가 자동으로 압축을 풀어서 읽어주기 때문에
    압축 파일(seoul.csv.gz)이든 일반 CSV(seoul.csv)든 코드 변경 없이 그대로 동작합니다."""
    raw = pd.read_csv(csv_url, dtype=str)
    raw = raw.rename(columns=CSV_COLUMN_TO_API)
    return postprocess_df(raw)


# --------------------------------------------------------------------
# 5) 사이드바 - 데이터 불러오기 방식 선택
# --------------------------------------------------------------------
st.sidebar.title("🔎 데이터 불러오기")

# GitHub 웹 업로드는 25MB 제한이 있어서, gzip으로 압축한 .csv.gz 파일을 기본값으로 씁니다.
DEFAULT_CSV_URL = "https://raw.githubusercontent.com/ok4506-bot/my-house-price/main/seoul.csv.gz"

data_source = st.sidebar.radio(
    "데이터 불러오기 방식",
    ["📄 CSV 파일 (빠름)", "🌐 Open API 직접 호출 (느림)"],
    help="CSV 모드는 미리 colab_fetch_seoul_csv.py로 받아둔 파일을 그대로 읽어서 훨씬 빠릅니다. "
         "API 모드는 그때그때 서울 열린데이터광장에서 직접 받아오기 때문에 2006~2026년 전체를 "
         "조회하면 시간이 오래 걸립니다.",
)

if data_source == "📄 CSV 파일 (빠름)":
    csv_url = DEFAULT_CSV_URL  # 주소는 고정 - 사이드바에서 직접 바꿔 입력하지 않습니다.
    if st.sidebar.button("🔄 새로고침 (캐시 지우고 다시 불러오기)", use_container_width=True):
        st.cache_data.clear()
    with st.spinner("CSV 파일을 불러오는 중..."):
        try:
            raw_df = csv_to_df(csv_url)
            fetch_errors, gu_last_error = [], {}
        except Exception as e:
            st.error(
                f"🚨 CSV 파일을 불러오지 못했어요: {e}\n\n"
                "주소가 올바른지, 파일이 실제로 그 경로에 있는지 확인해주세요. "
                "(colab_fetch_seoul_csv.py를 먼저 실행해서 seoul.csv를 만든 뒤, "
                "GitHub 등에 업로드했는지 확인하세요.)"
            )
            st.stop()
else:
    # API 모드에서만 인증키가 필요합니다.
    api_key = st.secrets.get("SEOUL_API_KEY", "") if hasattr(st, "secrets") else ""
    if not api_key:
        st.error(
            "🔑 서울 열린데이터광장 인증키가 설정되어 있지 않아요.\n\n"
            "앱을 배포한 곳의 **Settings → Secrets**에 아래처럼 추가해주세요.\n\n"
            "```toml\nSEOUL_API_KEY = \"발급받은_인증키\"\n```"
        )
        st.stop()

    gu_selected_api = st.sidebar.multiselect(
        "자치구 필터 (미선택 시 서울 전체 25개 구)",
        options=sorted(OFFICIAL_GU_CODE.keys()), default=[],
    )
    max_pages = st.sidebar.number_input(
        "연도·자치구 조합당 최대 페이지 수 (1페이지=최대 1,000건)",
        min_value=1, max_value=200, value=5,
        help="값이 클수록 학습 데이터가 많아지지만 (연도 수 × 자치구 수 × 이 값)만큼 API 호출이 늘어납니다.",
    )
    if st.sidebar.button("🔄 새로고침 (캐시 지우고 다시 조회)", use_container_width=True):
        st.cache_data.clear()

    if gu_selected_api:
        effective_gu_cd_list = [OFFICIAL_GU_CODE[nm] for nm in gu_selected_api]
    else:
        effective_gu_cd_list = list(OFFICIAL_GU_CODE.values())

    year_list_all = [str(y) for y in range(EARLIEST_YEAR, CURRENT_YEAR + 1)]
    with st.spinner("서울 열린데이터광장에서 실거래가 데이터를 불러오는 중... (시간이 꽤 걸릴 수 있어요)"):
        rows, fetch_errors, gu_last_error = fetch_all(
            api_key, tuple(year_list_all), tuple(effective_gu_cd_list), max_pages
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

# --------------------------------------------------------------------
# 5-1) 불러온 데이터에 적용하는 공통 필터 (CSV/API 어느 쪽으로 불러왔든 동일하게 동작)
# --------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📅 표시 범위 필터")
years = st.sidebar.slider(
    "계약연도 범위", min_value=EARLIEST_YEAR, max_value=CURRENT_YEAR,
    value=(EARLIEST_YEAR, CURRENT_YEAR),
)
gu_selected = st.sidebar.multiselect(
    "자치구 필터 (미선택 시 전체)", options=sorted(OFFICIAL_GU_CODE.keys()), default=[],
)

raw_df = raw_df[(raw_df["CTRT_YEAR"] >= years[0]) & (raw_df["CTRT_YEAR"] <= years[1])]
if gu_selected:
    raw_df = raw_df[raw_df["CGG_NM"].isin(gu_selected)]

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
# 8) 특정 조건(자치구·면적·층·연식) 예측가 — 가장 먼저 보여주는 핵심 기능
# --------------------------------------------------------------------
st.markdown("---")
st.subheader("🔮 특정 조건으로 아파트 가격 바로 예측하기")

with st.container(border=True):
    pick_gu3 = st.selectbox("자치구", sorted(OFFICIAL_GU_CODE.keys()), key="direct_gu")
    c1, c2, c3 = st.columns(3)
    with c1:
        area = st.slider("건물면적(㎡)", min_value=20.0, max_value=250.0, value=84.0, step=1.0)
    with c2:
        floor = st.slider("층", min_value=-3, max_value=50, value=10, step=1)
    with c3:
        age = st.slider("연식(건축 후 경과년수)", min_value=0, max_value=60, value=15, step=1)

    pred_eok = predict_price_eok(pick_gu3, area, floor, age)
    st.markdown(
        f"<div style='text-align:center; margin-top:1.0em;'>"
        f"<span style='font-size:1.1em;'>🔮 {pick_gu3} · {area:.0f}㎡ · {floor}층 · 연식 {age}년 예측 가격</span><br>"
        f"<span style='font-size:3.2em; font-weight:800; color:#e0522f;'>{pred_eok:.2f}억원</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------
# 9) R² 카드
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


# --------------------------------------------------------------------------------------
# 10) 자치구 · 법정동별 실거래가 추이 & 예측
#    (매물 하나하나가 아니라, 구/동 단위로 2006~2026년 전체 연도별 중위값과 추세선을 보여줍니다)
# --------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📈 자치구 · 법정동별 실거래가 추이 & 예측")
st.caption(
    "2006년부터 2026년까지 연도별 물건금액 중위값(억원)을 점으로 표시하고, "
    "옅은 막대로 그 해의 최소~최대 가격 범위를 함께 보여줍니다. "
    "실선(추세선)은 전체 기간에 걸친 로그 스케일 회귀선이고, 점선은 앞으로 몇 년을 더 예측한 값입니다."
)


def yearly_median_table(df_subset):
    """계약연도별 물건금액 중위값·최소·최대·거래건수를 구해서 표로 돌려줍니다. (실제 데이터가 있는 연도만)"""
    g = (
        df_subset.groupby("CTRT_YEAR")["PRICE_EOK"]
        .agg(median="median", min="min", max="max", count="count")
        .reset_index()
        .dropna(subset=["CTRT_YEAR", "median"])
    )
    g = g[(g["CTRT_YEAR"] >= EARLIEST_YEAR) & (g["CTRT_YEAR"] <= CURRENT_YEAR)]
    return g.sort_values("CTRT_YEAR")


def fit_year_trend(yearly_df):
    """연도(x) -> log(중위가격)(y)로 선형회귀 추세선을 학습합니다."""
    if len(yearly_df) < 2:
        return None, None
    X_year = yearly_df[["CTRT_YEAR"]].to_numpy(dtype=float)
    y_log_price = np.log(yearly_df["median"].to_numpy(dtype=float))
    trend_model = LinearRegression()
    trend_model.fit(X_year, y_log_price)
    r2_in_sample = r2_score(y_log_price, trend_model.predict(X_year))
    return trend_model, r2_in_sample


def trend_line_series(trend_model, start_year, end_year):
    """trend_model로 start_year부터 end_year까지 매년 예측 가격(억원)을 만들어줍니다."""
    yrs = list(range(start_year, end_year + 1))
    log_price = trend_model.predict(np.array(yrs, dtype=float).reshape(-1, 1))
    return yrs, np.exp(log_price)


def compute_region_highlights(sub_df, yearly_df):
    """이 지역에서 가장 비싼 주택, 가장 싼 주택, 그리고 지역 가격 흐름을
    가장 잘 따라가는(연도별 지역 중위가와 가장 차이가 적은) 대표 주택을 찾습니다."""
    sub_df = sub_df.dropna(subset=["PRICE_EOK"])
    if sub_df.empty:
        return None

    max_row = sub_df.loc[sub_df["PRICE_EOK"].idxmax()]
    min_row = sub_df.loc[sub_df["PRICE_EOK"].idxmin()]

    rep_name, rep_note = None, ""
    if not yearly_df.empty:
        year_to_median = dict(zip(yearly_df["CTRT_YEAR"], yearly_df["median"]))
        tmp = sub_df.dropna(subset=["BLDG_NM", "CTRT_YEAR"]).copy()
        tmp["지역중위가"] = tmp["CTRT_YEAR"].map(year_to_median)
        tmp = tmp.dropna(subset=["지역중위가"])
        tmp = tmp[tmp["BLDG_NM"].str.strip() != ""]
        if not tmp.empty:
            # 자신이 거래된 해의 지역 중위가격과 (로그 스케일로) 얼마나 차이나는지 평균 내서,
            # 그 차이가 가장 작은 건물을 "지역 추세를 가장 잘 반영하는 대표 주택"으로 선정
            tmp["편차"] = (np.log(tmp["PRICE_EOK"]) - np.log(tmp["지역중위가"])).abs()
            building_dev = tmp.groupby("BLDG_NM")["편차"].mean().sort_values()
            if len(building_dev):
                rep_name = building_dev.index[0]
                rep_note = f"(지역 중위가격과의 평균 차이: {building_dev.iloc[0]*100:.1f}%)"

    return {
        "max_name": (max_row.get("BLDG_NM") or "").strip() or "(건물명 미상)",
        "max_price": max_row["PRICE_EOK"],
        "min_name": (min_row.get("BLDG_NM") or "").strip() or "(건물명 미상)",
        "min_price": min_row["PRICE_EOK"],
        "rep_name": rep_name or "정보 부족",
        "rep_note": rep_note,
    }


view_unit = st.segmented_control(
    "보기 단위", options=["자치구별", "법정동별"], default="자치구별",
)
if view_unit is None:  # 혹시 아무것도 선택 안 된 상태가 되면 기본값으로 되돌립니다
    view_unit = "자치구별"

if view_unit == "자치구별":
    gu_options = sorted(apt_df["CGG_NM"].unique())  # 서울시 전체 자치구
    regions_selected = st.multiselect(
        "비교할 자치구 (기본으로 서울 전체가 선택되어 있어요 — 하나씩 클릭해서 빼면 그래프에서 제외돼요)",
        options=gu_options, default=gu_options,
    )
    region_frames = {nm: apt_df[apt_df["CGG_NM"] == nm] for nm in regions_selected}
else:
    pick_gu2 = st.selectbox("자치구", sorted(apt_df["CGG_NM"].unique()), key="trend_gu")
    dong_options2 = sorted(apt_df.loc[apt_df["CGG_NM"] == pick_gu2, "STDG_NM"].dropna().unique())
    default_dong = dong_options2[:2] if len(dong_options2) >= 2 else dong_options2
    regions_selected = st.multiselect(
        f"비교할 {pick_gu2}의 법정동을 선택하세요 (최대 5개)",
        options=dong_options2, default=default_dong, max_selections=5,
    )
    region_frames = {
        nm: apt_df[(apt_df["CGG_NM"] == pick_gu2) & (apt_df["STDG_NM"] == nm)] for nm in regions_selected
    }

horizon = st.slider("몇 년 뒤까지 예측할까요?", min_value=1, max_value=5, value=5)

if len(regions_selected) > 8:
    st.caption(
        f"ℹ️ 지금 {len(regions_selected)}개 지역을 한번에 표시하고 있어요. 선이 너무 많아 복잡하면, "
        "위 목록에서 몇 개를 클릭해서 하나씩 빼보세요."
    )

if not regions_selected:
    st.info("위에서 비교하고 싶은 자치구(또는 법정동)를 1개 이상 선택해주세요.")
else:
    chart_rows = []
    yearly_by_region = {}
    trend_r2_by_region = {}

    for region_name, sub_df in region_frames.items():
        # 2006~2026년 전체 구간을 다 보여주기 위해, 실제 데이터가 있는 연도는 "실제" 점으로 표시
        yearly_df = yearly_median_table(sub_df)
        yearly_by_region[region_name] = yearly_df
        for _, r in yearly_df.iterrows():
            chart_rows.append({
                "지역": region_name, "연도": int(r["CTRT_YEAR"]),
                "가격(억원)": r["median"], "구분": "실제 중위가격", "거래건수": int(r["count"]),
            })

        trend_model, r2_trend = fit_year_trend(yearly_df)
        if trend_model is None:
            st.caption(f"⚠️ **{region_name}**: 연도별 데이터 포인트가 부족해 추세선을 만들 수 없어요.")
            continue
        trend_r2_by_region[region_name] = r2_trend

        first_year = int(yearly_df["CTRT_YEAR"].min())
        last_year = int(yearly_df["CTRT_YEAR"].max())

        # 추세선(실선): 실제 데이터가 있는 첫 해 ~ 마지막 해까지, 2006~2026년 전체를 아우르도록 계산
        fit_years, fit_prices = trend_line_series(trend_model, min(EARLIEST_YEAR, first_year), last_year)
        for yy, pp in zip(fit_years, fit_prices):
            chart_rows.append({"지역": region_name, "연도": yy, "가격(억원)": pp,
                                "구분": "추세선(적합)", "거래건수": None})

        # 추세선(점선): 마지막 실제 연도부터 미래 예측 구간까지
        fc_years, fc_prices = trend_line_series(trend_model, last_year, last_year + horizon)
        for yy, pp in zip(fc_years, fc_prices):
            chart_rows.append({"지역": region_name, "연도": yy, "가격(억원)": pp,
                                "구분": "추세선(예측)", "거래건수": None})

    if chart_rows:
        chart_df = pd.DataFrame(chart_rows)
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, region_name in enumerate(regions_selected):
            color = colors[i % len(colors)]
            reg_df = chart_df[chart_df["지역"] == region_name]

            # 연도별 최소~최대 가격 범위를 옅은 막대로 먼저 그려서 (실제 점/추세선보다 뒤에 오도록)
            range_df = yearly_by_region.get(region_name)
            if range_df is not None and not range_df.empty:
                fig.add_trace(go.Bar(
                    x=range_df["CTRT_YEAR"], y=range_df["max"] - range_df["min"], base=range_df["min"],
                    name=f"{region_name}·최소~최대 범위", marker=dict(color=color), opacity=0.22,
                    legendgroup=region_name, showlegend=False, width=0.6,
                    customdata=np.stack([range_df["min"], range_df["max"]], axis=-1),
                    hovertemplate=(
                        f"<b>{region_name}</b><br>연도: %{{x}}<br>"
                        "최소가격: %{customdata[0]:.2f}억원<br>최고가격: %{customdata[1]:.2f}억원"
                        "<extra></extra>"
                    ),
                ))

            actual = reg_df[reg_df["구분"] == "실제 중위가격"].sort_values("연도")
            fig.add_trace(go.Scatter(
                x=actual["연도"], y=actual["가격(억원)"], mode="markers",
                name=f"{region_name}·실제 중위가격",
                marker=dict(size=9, color=color), legendgroup=region_name,
                customdata=actual["거래건수"],
                hovertemplate=(
                    f"<b>{region_name}</b><br>연도: %{{x}}<br>"
                    "실제 중위가격: %{y:.2f}억원<br>거래건수: %{customdata}건"
                    "<extra></extra>"
                ),
            ))

            fit_line = reg_df[reg_df["구분"] == "추세선(적합)"].sort_values("연도")
            fig.add_trace(go.Scatter(
                x=fit_line["연도"], y=fit_line["가격(억원)"], mode="lines",
                name=f"{region_name}·추세선(계산값)",
                line=dict(color=color, width=2), legendgroup=region_name,
                hovertemplate=(
                    f"<b>{region_name}</b><br>연도: %{{x}}<br>"
                    "추세선 계산값(실제 데이터 아님): %{y:.2f}억원<extra></extra>"
                ),
            ))

            fc_line = reg_df[reg_df["구분"] == "추세선(예측)"].sort_values("연도")
            fig.add_trace(go.Scatter(
                x=fc_line["연도"], y=fc_line["가격(억원)"], mode="lines",
                name=f"{region_name}·미래 예측값",
                line=dict(color=color, width=2, dash="dash"), legendgroup=region_name,
                hovertemplate=(
                    f"<b>{region_name}</b><br>연도: %{{x}}<br>"
                    "미래 예측값(추세선 연장): %{y:.2f}억원<extra></extra>"
                ),
            ))

        fig.update_layout(
            height=520, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="연도", yaxis_title="중위가격(억원)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "📌 그래프의 **점(실제 중위가격)**은 특정 아파트 한 채의 가격이 아니라, "
            "그 해에 해당 지역에서 거래된 모든 아파트 가격의 **중위값(중간값)**입니다. "
            "옅은 막대는 그 해의 **최소~최대 가격 범위**를 보여줍니다 (마우스를 올리면 정확한 값이 나와요). "
            "실선(추세선)과 점선(미래 예측값)은 중위값을 바탕으로 계산한 수치이며, 실제로 거래된 값이 아닙니다."
        )

        # --- 그래프를 "눌러본" 것처럼, 지역을 선택하면 상세 정보를 보여주는 부분 ---
        st.markdown("##### 👇 지역을 선택해서 상세 정보 보기 (그래프에서 눌러보는 것과 같아요)")
        detail_region = st.selectbox("상세히 볼 지역", options=regions_selected, key="detail_region")
        highlight = compute_region_highlights(region_frames[detail_region], yearly_by_region[detail_region])

        if highlight is None:
            st.info(f"{detail_region}에는 표시할 거래 데이터가 없습니다.")
        else:
            hc1, hc2, hc3 = st.columns(3)
            with hc1:
                st.metric("🔺 최고가 주택", highlight["max_name"], f"{highlight['max_price']:.2f}억원")
            with hc2:
                st.metric("🔻 최저가 주택", highlight["min_name"], f"{highlight['min_price']:.2f}억원")
            with hc3:
                st.metric("📌 지역 추세 대표 주택", highlight["rep_name"])
                if highlight["rep_note"]:
                    st.caption(highlight["rep_note"])
            if detail_region in trend_r2_by_region:
                st.caption(f"{detail_region} 추세선 적합도(R²): {trend_r2_by_region[detail_region]:.3f}")
    else:
        st.warning("선택한 지역에 표시할 데이터가 없습니다.")

st.markdown("---")
st.caption(
    "회귀 모델은 자치구·건물면적·층·건축년도(연식) 네 가지만 사용한 단순 선형회귀이며, "
    "실제 시세와는 차이가 있을 수 있습니다. 물건금액은 로그 변환 후 학습했고, "
    "예측값은 다시 지수변환(exp)해서 원래 크기(억원)로 보여드립니다. "
    "연도별 추세 예측 역시 과거 흐름을 단순 직선으로 연장한 참고용 수치이며, "
    "실제 시장은 정책·금리 등 다양한 변수에 영향을 받습니다."
)
