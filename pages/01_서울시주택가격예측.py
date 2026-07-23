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
    df["CTRT_YEAR"] = df["CTRT_DATE"].dt.year  # 계약연도 (실제 가격 추이 그래프용)
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


# --------------------------------------------------------------------------------------
# 9) 자치구 · 법정동별 실거래가 추이 & 예측
#    (매물 하나하나가 아니라, 구/동 단위로 연도별 중위값 추이를 보고 미래를 예측합니다)
# --------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📈 자치구 · 법정동별 실거래가 추이 & 예측")
st.caption(
    "연도별 물건금액 중위값(억원)의 실제 흐름을 보여주고, 간단한 선형 추세선으로 "
    "앞으로 몇 년의 가격을 함께 예측합니다. (실제=실선, 예측=점선)"
)


def yearly_median_table(df_subset):
    """계약연도별 물건금액 중위값·거래건수를 구해서 표로 돌려줍니다."""
    g = (
        df_subset.groupby("CTRT_YEAR")["PRICE_EOK"]
        .agg(median="median", count="count")
        .reset_index()
        .dropna(subset=["CTRT_YEAR", "median"])
    )
    g = g[(g["CTRT_YEAR"] >= EARLIEST_YEAR) & (g["CTRT_YEAR"] <= CURRENT_YEAR)]
    return g.sort_values("CTRT_YEAR")


def fit_year_trend_and_forecast(yearly_df, horizon):
    """연도(x) -> log(중위가격)(y)로 아주 단순한 선형회귀 추세선을 만들고,
    마지막 실제 연도 다음부터 horizon년 만큼 미래 가격을 예측합니다."""
    if len(yearly_df) < 2:
        return None  # 추세선을 그리기엔 실제 연도별 데이터 포인트가 너무 적음

    X_year = yearly_df[["CTRT_YEAR"]].to_numpy(dtype=float)
    y_log_price = np.log(yearly_df["median"].to_numpy(dtype=float))

    trend_model = LinearRegression()
    trend_model.fit(X_year, y_log_price)
    r2_in_sample = r2_score(y_log_price, trend_model.predict(X_year))

    last_year = int(yearly_df["CTRT_YEAR"].max())
    future_years = list(range(last_year + 1, last_year + 1 + horizon))
    future_log_price = trend_model.predict(np.array(future_years, dtype=float).reshape(-1, 1))
    future_price_eok = np.exp(future_log_price)

    return {
        "last_year": last_year,
        "future_years": future_years,
        "future_price_eok": future_price_eok,
        "r2": r2_in_sample,
    }


view_unit = st.radio("보기 단위", ["자치구별", "법정동별"], horizontal=True)

if view_unit == "자치구별":
    gu_options = sorted(apt_df["CGG_NM"].unique())
    default_gu = gu_options[:2] if len(gu_options) >= 2 else gu_options
    regions_selected = st.multiselect(
        "비교할 자치구를 선택하세요 (최대 5개)", options=gu_options, default=default_gu, max_selections=5,
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

horizon = st.slider("몇 년 뒤까지 예측할까요?", min_value=1, max_value=5, value=3)

if not regions_selected:
    st.info("위에서 비교하고 싶은 자치구(또는 법정동)를 1개 이상 선택해주세요.")
else:
    chart_rows = []
    forecast_rows = []
    for region_name, sub_df in region_frames.items():
        yearly_df = yearly_median_table(sub_df)
        for _, r in yearly_df.iterrows():
            chart_rows.append({
                "지역": region_name, "연도": int(r["CTRT_YEAR"]),
                "중위가(억원)": r["median"], "구분": "실제", "거래건수": int(r["count"]),
            })

        result = fit_year_trend_and_forecast(yearly_df, horizon)
        if result is None:
            st.caption(f"⚠️ **{region_name}**: 연도별 데이터 포인트가 부족해 추세 예측을 만들 수 없어요.")
            continue

        # 예측 점선이 실제 실선과 끊기지 않고 이어지도록, 마지막 실제 연도 값을 예측 계열에도 하나 넣어줌
        last_row = yearly_df[yearly_df["CTRT_YEAR"] == result["last_year"]].iloc[0]
        chart_rows.append({
            "지역": region_name, "연도": result["last_year"],
            "중위가(억원)": last_row["median"], "구분": "예측", "거래건수": int(last_row["count"]),
        })
        for fy, fp in zip(result["future_years"], result["future_price_eok"]):
            chart_rows.append({"지역": region_name, "연도": fy, "중위가(억원)": fp, "구분": "예측", "거래건수": None})
            forecast_rows.append({
                "지역": region_name, "연도": fy, "예측 중위가(억원)": round(float(fp), 2),
                "추세선 적합도(R²)": round(result["r2"], 3),
            })

    if chart_rows:
        chart_df = pd.DataFrame(chart_rows)
        fig = px.line(
            chart_df, x="연도", y="중위가(억원)", color="지역", line_dash="구분",
            markers=True, hover_data={"거래건수": True},
            labels={"중위가(억원)": "중위가격(억원)"},
        )
        fig.update_traces(selector=dict(mode="markers+lines"))
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        if forecast_rows:
            st.write("**연도별 예측 중위가격**")
            st.dataframe(pd.DataFrame(forecast_rows), use_container_width=True, hide_index=True)
            st.caption(
                "추세선 적합도(R²)는 이 지역의 연도별 실제 중위가격이 직선(로그 스케일) 추세와 "
                "얼마나 잘 맞는지를 나타냅니다. 거래건수가 적은 연도가 있으면 중위가격이 들쭉날쭉해서 "
                "예측이 부정확할 수 있어요."
            )
    else:
        st.warning("선택한 지역에 표시할 데이터가 없습니다.")


# --------------------------------------------------------------------------------------
# 10) (보너스) 대표 매물 조건으로 모델 예측가 보기
#     - 위에서 학습한 자치구·면적·층·연식 회귀 모델을 그대로 활용합니다.
# --------------------------------------------------------------------------------------
with st.expander("🔧 특정 조건(자치구·면적·층·연식)의 대표 매물 예측가 보기"):
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

st.markdown("---")
st.caption(
    "회귀 모델은 자치구·건물면적·층·건축년도(연식) 네 가지만 사용한 단순 선형회귀이며, "
    "실제 시세와는 차이가 있을 수 있습니다. 물건금액은 로그 변환 후 학습했고, "
    "예측값은 다시 지수변환(exp)해서 원래 크기(억원)로 보여드립니다. "
    "연도별 추세 예측 역시 과거 흐름을 단순 직선으로 연장한 참고용 수치이며, "
    "실제 시장은 정책·금리 등 다양한 변수에 영향을 받습니다."
)
