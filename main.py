# -*- coding: utf-8 -*-
"""
서울시 부동산 실거래가 대시보드
- 데이터 출처: 서울 열린데이터광장 Open API (tbLnOpendataRtmsV)
- 서울 열린데이터광장: https://data.seoul.go.kr
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# --------------------------------------------------------------------------------------
# 기본 설정
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="서울시 부동산 실거래가 대시보드",
    page_icon="🏠",
    layout="wide",
)

BASE_URL = "http://openapi.seoul.go.kr:8088"
SERVICE = "tbLnOpendataRtmsV"
MAX_PER_CALL = 1000
APP_DIR = Path(__file__).parent
GEOJSON_PATH = APP_DIR / "seoul_gu.geojson"
CURRENT_YEAR = datetime.now().year
EARLIEST_YEAR = 2006  # 실거래가 신고제도 시행연도(대략)
BLDG_USG_OPTIONS = ["아파트", "단독다가구", "연립다세대", "오피스텔"]

# 서울시 25개 자치구의 "법정동코드 기준 시군구코드"입니다.
# ⚠️ 주의: seoul_gu.geojson 파일 안의 code 속성은 통계청(KOSTAT) 내부 일련번호라서
#     서울 열린데이터광장 API가 쓰는 CGG_CD(법정동코드)와 값이 다릅니다.
#     (예: geojson의 강동구 code='11250'이지만, 실제 API의 강동구 CGG_CD는 '11740'.)
#     그래서 API 조회는 항상 아래의 공식 코드로 하고, 지도 매칭은 코드가 아닌
#     "자치구 이름" 문자열로 맞춥니다. 이 값들은 실제 API 응답 예시(은평구 11380,
#     강남구 11680, 구로구 11530)와도 일치하는 공식 코드입니다.
OFFICIAL_GU_CODE = {
    "종로구": "11110", "중구": "11140", "용산구": "11170", "성동구": "11200",
    "광진구": "11215", "동대문구": "11230", "중랑구": "11260", "성북구": "11290",
    "강북구": "11305", "도봉구": "11320", "노원구": "11350", "은평구": "11380",
    "서대문구": "11410", "마포구": "11440", "양천구": "11470", "강서구": "11500",
    "구로구": "11530", "금천구": "11545", "영등포구": "11560", "동작구": "11590",
    "관악구": "11620", "서초구": "11650", "강남구": "11680", "송파구": "11710",
    "강동구": "11740",
}

COLUMN_TYPES = {
    "RCPT_YR": "str", "CGG_CD": "str", "CGG_NM": "str", "STDG_CD": "str", "STDG_NM": "str",
    "LOTNO_SE": "str", "LOTNO_SE_NM": "str", "MNO": "str", "SNO": "str", "BLDG_NM": "str",
    "CTRT_DAY": "str", "THING_AMT": "float", "ARCH_AREA": "float", "LAND_AREA": "float",
    "FLR": "float", "RGHT_SE": "str", "RTRCN_DAY": "str", "ARCH_YR": "float",
    "BLDG_USG": "str", "DCLR_SE": "str", "OPBIZ_RESTAGNT_SGG_NM": "str",
}



# --------------------------------------------------------------------------------------
# API 호출 관련 함수
# --------------------------------------------------------------------------------------
def build_url(api_key, start, end, rcpt_yr="", cgg_cd=""):
    """포지셔널 옵션 인자 중 앞부분(RCPT_YR, CGG_CD)만 사용, 뒤는 생략.
    CGG_NM 대신 CGG_CD를 쓰는 이유: RCPT_YR 바로 다음 자리라 중간에 빈 칸이
    생기지 않아 자치구 필터가 훨씬 안정적으로 동작합니다."""
    parts = [api_key, "json", SERVICE, str(start), str(end)]
    tail = [str(rcpt_yr), str(cgg_cd)]
    while tail and tail[-1] == "":
        tail.pop()
    parts += tail
    return BASE_URL + "/" + "/".join(parts)


def _parse_response(data):
    """API 응답에서 (rows, total_count, error_message)를 반환."""
    if SERVICE not in data:
        # 상위 레벨에 에러 정보가 있는 경우
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


def fetch_year(api_key, year, cgg_cd, max_pages, progress_cb=None):
    """특정 연도(+구) 데이터를 페이지네이션으로 모두 가져옴."""
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

        rows, total_count_new, err = _parse_response(data)
        if err:
            # 해당 구간에 결과가 없는 경우(ERROR-200 등)도 err로 오므로 조용히 종료
            if "ERROR-200" in err or "결과값이 없습니다" in err or "데이터가 없습니다" in err:
                break
            return rows_all, total_count, err

        if total_count_new is not None:
            total_count = total_count_new
        rows_all.extend(rows)
        page += 1
        if progress_cb:
            progress_cb(year, cgg_cd, page, len(rows_all), total_count)

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
    """연도 x 자치구코드(CGG_CD) 조합으로 데이터를 모아 리스트로 반환.
    반환값: (전체 행 리스트, 에러 메시지 리스트, 자치구코드별 마지막 에러 메시지 dict)
    - 자치구별 에러를 따로 남겨서, "거래가 0건"이 진짜 데이터가 없는 건지
      아니면 API 호출이 실패한 건지 화면에서 바로 구분할 수 있게 합니다.
    """
    all_rows = []
    combos = [(y, g) for y in years for g in (gu_cd_list if gu_cd_list else [""])]
    errors = []
    gu_last_error = {}  # {자치구코드: "마지막 에러 메시지"} - 실제로 API 호출이 실패했던 구만 기록
    prog = st.progress(0.0, text="데이터 수집 준비 중...")
    n = len(combos)
    for i, (y, g) in enumerate(combos):
        label = f"{y}년" + (f" · 자치구코드 {g}" if g else " · 서울 전체")
        prog.progress((i) / max(n, 1), text=f"수집 중: {label} ({i+1}/{n})")

        rows, total, err = fetch_year(api_key, y, g, max_pages_per_combo)
        if err:
            # 일시적인 오류(트래픽 제한, 서버 지연 등)일 수 있으니 잠깐 쉬었다가 한 번 더 시도
            time.sleep(1.0)
            rows, total, err = fetch_year(api_key, y, g, max_pages_per_combo)

        if err:
            errors.append(f"{label}: {err}")
            if g:
                gu_last_error[g] = err
        all_rows.extend(rows)
        time.sleep(0.1)  # 자치구별 호출 사이에 짧게 쉬어서 순간적인 트래픽 제한을 피함
    prog.progress(1.0, text="수집 완료")
    time.sleep(0.2)
    prog.empty()
    return all_rows, errors, gu_last_error


def rows_to_df(rows):
    if not rows:
        return pd.DataFrame(columns=list(COLUMN_TYPES.keys()))
    df = pd.DataFrame(rows)
    for col, typ in COLUMN_TYPES.items():
        if col not in df.columns:
            df[col] = np.nan
        if typ == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["CTRT_DATE"] = pd.to_datetime(df["CTRT_DAY"], format="%Y%m%d", errors="coerce")
    df["CTRT_YEAR"] = df["CTRT_DATE"].dt.year
    df["PRICE_EOK"] = df["THING_AMT"] / 10000.0  # 만원 -> 억원
    # 취소된 거래 제외 (RTRCN_DAY가 채워져 있으면 취소건)
    df["IS_CANCELLED"] = df["RTRCN_DAY"].fillna("").astype(str).str.strip() != ""
    return df


@st.cache_data(show_spinner=False)
def load_gu_geojson():
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# 인증키 (Secrets에서만 로드 - 사이드바에 노출하지 않음)
# --------------------------------------------------------------------------------------
api_key = st.secrets.get("SEOUL_API_KEY", "") if hasattr(st, "secrets") else ""

st.title("🏠 서울시 부동산 실거래가 대시보드")
st.caption("데이터 출처: 서울 열린데이터광장 Open API — 부동산 실거래가 정보(tbLnOpendataRtmsV)")

if not api_key:
    st.error(
        "서울 열린데이터광장 인증키가 설정되어 있지 않습니다. "
        "Streamlit Cloud의 **Settings → Secrets**에 아래와 같이 추가해주세요.\n\n"
        "```toml\nSEOUL_API_KEY = \"발급받은_인증키\"\n```"
    )
    st.stop()

# --------------------------------------------------------------------------------------
# 사이드바 - 조회 조건 (인증키는 Secrets 사용, 화면에 노출하지 않음)
# --------------------------------------------------------------------------------------
st.sidebar.title("🔎 조회 조건")

years = st.sidebar.slider(
    "계약연도(접수연도) 범위",
    min_value=EARLIEST_YEAR, max_value=CURRENT_YEAR,
    value=(CURRENT_YEAR - 2, CURRENT_YEAR),
    help="가장 오래된 시기부터 지금까지 전체를 조회할 수도 있지만, API는 1회 호출당 최대 1,000건으로 제한되어 있어 "
         "범위가 넓을수록 호출 횟수와 시간이 크게 늘어납니다.",
)

gu_name_to_code = OFFICIAL_GU_CODE
all_gu_names = sorted(gu_name_to_code.keys())
gu_selected = st.sidebar.multiselect("자치구 필터 (미선택 시 서울 전체)", options=all_gu_names, default=[])

max_pages = st.sidebar.number_input(
    "연도·자치구 조합당 최대 페이지 수 (1페이지=최대 1,000건)",
    min_value=1, max_value=500, value=3,
    help="자치구를 선택하지 않으면 서울시 25개 자치구를 각각 따로 조회하여 특정 구가 누락되지 않도록 합니다. "
         "값이 클수록 더 많은 데이터를 가져오지만 (연도 수 × 자치구 수 × 이 값)만큼 API 호출이 늘어납니다.",
)

st.sidebar.caption(
    "⚠️ 서울 열린데이터광장 API는 1회 요청당 최대 1,000건, 일반 인증키 기준 일일 호출 횟수 제한이 있습니다. "
    "자치구를 선택하지 않으면 25개 구를 각각 조회하므로 호출 수가 (연도 수 × 25 × 페이지 수)만큼 늘어납니다. "
    "기간이 넓다면 페이지 수를 낮추거나 자치구를 좁혀서 사용하세요."
)

if st.sidebar.button("🔄 새로고침 (캐시 지우고 다시 조회)", use_container_width=True):
    st.cache_data.clear()

# 자치구 필터는 화면엔 이름(예: 강남구)으로 보여주지만, API 호출은 코드(CGG_CD)로 합니다.
# (이름을 그대로 쓰면 RCPT_YR과 CGG_NM 사이에 빈 칸이 생겨 필터가 제대로 안 먹는 문제가 있었습니다.)
if gu_selected:
    effective_gu_cd_list = [gu_name_to_code[nm] for nm in gu_selected]
else:
    effective_gu_cd_list = list(gu_name_to_code.values())

year_list = [str(y) for y in range(years[0], years[1] + 1)]
with st.spinner("서울 열린데이터광장에서 실거래가 데이터를 불러오는 중... (자치구별로 나눠서 조회합니다)"):
    rows, fetch_errors, gu_last_error = fetch_all(api_key, tuple(year_list), tuple(effective_gu_cd_list), max_pages)
df = rows_to_df(rows)

code_to_name = {v: k for k, v in OFFICIAL_GU_CODE.items()}
gu_error_by_name = {code_to_name.get(cd, cd): err for cd, err in gu_last_error.items()}

if gu_last_error:
    failed_names = ", ".join(sorted(code_to_name.get(cd, cd) for cd in gu_last_error))
    st.warning(
        f"⚠️ 아래 자치구는 실제로 거래가 없는 게 아니라 **API 호출 자체가 실패**해서 0건으로 보였을 수 있습니다: "
        f"**{failed_names}**\n\n"
        "인증키의 일일 호출 한도를 초과했거나, 순간적으로 서버 응답이 지연됐을 가능성이 있습니다. "
        "아래 오류 상세를 확인하고, 잠시 후 '🔄 새로고침' 버튼으로 다시 시도해보세요."
    )

if fetch_errors:
    with st.expander(f"⚠️ 수집 중 오류 발생 ({len(fetch_errors)}건) - 클릭해서 자세히 보기", expanded=bool(gu_last_error)):
        for e in fetch_errors:
            st.write("- ", e)

if df.empty:
    st.warning("조회된 데이터가 없습니다. 조건을 조정한 뒤 다시 시도해주세요.")
    st.stop()


# 취소 거래 제외 옵션
exclude_cancelled = st.sidebar.checkbox("취소된 거래 제외", value=True)
work_df = df[~df["IS_CANCELLED"]].copy() if exclude_cancelled else df.copy()
work_df = work_df.dropna(subset=["THING_AMT"])

# 건물용도 필터 (분석 화면 공통)
st.sidebar.markdown("---")
usg_present = [u for u in BLDG_USG_OPTIONS if u in work_df["BLDG_USG"].unique()] or sorted(work_df["BLDG_USG"].dropna().unique())
usg_selected = st.sidebar.multiselect("건물용도 필터 (분석 화면 공통 적용)", options=usg_present, default=usg_present)
if usg_selected:
    work_df = work_df[work_df["BLDG_USG"].isin(usg_selected)]

# --------------------------------------------------------------------------------------
# 상단 요약 지표
# --------------------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("총 거래건수", f"{len(work_df):,} 건")
c2.metric("중위 거래가격", f"{work_df['PRICE_EOK'].median():.2f} 억원")
c3.metric("평균 거래가격", f"{work_df['PRICE_EOK'].mean():.2f} 억원")
c4.metric("최고가", f"{work_df['PRICE_EOK'].max():.2f} 억원")
c5.metric("자치구 수", f"{work_df['CGG_NM'].nunique()} 개")

st.markdown("---")

tab_map_gu, tab_year, tab_usg, tab_floor, tab_narrative, tab_raw = st.tabs(
    ["🗺️ 자치구별 지도 · 순위", "🏗️ 건축년도별 가격", "🏢 건물용도별 가격",
     "🏬 층-가격 관계", "📅 연도별 특징", "📄 원본 데이터"]
)

# --------------------------------------------------------------------------------------
# 1) 자치구별 중위가격 지도 + 상위/하위 5개 순위표
#    (법정동 단위 집계는 빼고, 자치구 단위로만 집계 → 외부 좌표 조회 없이 즉시 계산됩니다)
# --------------------------------------------------------------------------------------
with tab_map_gu:
    st.subheader("자치구(CGG_NM)별 물건금액 지도 · 순위")

    geojson = load_gu_geojson()
    all_gu_df = pd.DataFrame({"CGG_NM": sorted(OFFICIAL_GU_CODE.keys())})

    computed = (
        work_df.groupby("CGG_NM")["PRICE_EOK"]
        .agg(median="median", count="count", mean="mean", max="max", min="min")
        .reset_index()
    )
    # 서울시 25개 자치구 전체를 항상 표시 (데이터가 없는 구는 결측으로 표시)
    # ⚠️ geojson의 code 속성이 API의 CGG_CD와 값이 달라서(주석 참고), 코드가 아니라
    #    "자치구 이름" 문자열 기준으로 지도/표를 맞춥니다.
    gu_stat = all_gu_df.merge(computed[["CGG_NM", "median", "count", "mean", "min", "max"]],
                               on="CGG_NM", how="left")
    gu_stat["count"] = gu_stat["count"].fillna(0).astype(int)
    gu_stat["rank_median"] = gu_stat["median"].rank(ascending=False, method="min")
    # 거래 0건이 "진짜 데이터 없음"인지 "API 호출 실패"인지 구분해서 표에 남깁니다.
    gu_stat["비고"] = gu_stat["CGG_NM"].map(
        lambda nm: "⚠️ 조회 실패(재시도 필요)" if nm in gu_error_by_name else ""
    )

    n_missing = gu_stat["median"].isna().sum()
    if n_missing:
        st.caption(f"※ {n_missing}개 자치구는 현재 조건에서 거래 데이터가 없어 회색으로 표시됩니다.")

    fig = px.choropleth_mapbox(
        gu_stat, geojson=geojson, locations="CGG_NM", color="median",
        featureidkey="properties.name",
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        zoom=9.5, center={"lat": 37.5546, "lon": 126.9706},
        opacity=0.8,
        hover_name="CGG_NM",
        hover_data={"rank_median": True, "median": ":.2f", "mean": ":.2f",
                    "max": ":.2f", "min": ":.2f", "count": True},
        labels={"median": "중위가격(억원)", "rank_median": "중위가 순위"},
    )
    fig.update_traces(marker_line_width=1, marker_line_color="white")
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0),
                       coloraxis_colorbar=dict(title="중위가격(억원)"))
    st.plotly_chart(fig, use_container_width=True)

    ranked = gu_stat.dropna(subset=["median"]).copy()

    def show_top_bottom(col, label, fmt="{:.2f}"):
        """col 기준으로 상위 5개 / 하위 5개 자치구 표를 나란히 보여주는 도우미 함수."""
        st.markdown(f"**{label} 기준 자치구 순위**")
        c1, c2 = st.columns(2)
        sorted_df = ranked.sort_values(col, ascending=False)
        with c1:
            st.caption(f"🔼 {label} 상위 5개 자치구")
            top5 = sorted_df.head(5)[["CGG_NM", col, "count"]].rename(
                columns={"CGG_NM": "자치구", col: label, "count": "거래건수"})
            st.dataframe(top5.style.format({label: fmt}), hide_index=True, use_container_width=True)
        with c2:
            st.caption(f"🔽 {label} 하위 5개 자치구")
            bottom5 = sorted_df.tail(5).sort_values(col, ascending=True)[["CGG_NM", col, "count"]].rename(
                columns={"CGG_NM": "자치구", col: label, "count": "거래건수"})
            st.dataframe(bottom5.style.format({label: fmt}), hide_index=True, use_container_width=True)

    show_top_bottom("median", "중위값(억원)")
    show_top_bottom("max", "최댓값(억원)")
    show_top_bottom("min", "최솟값(억원)")

    with st.expander("서울시 25개 자치구 전체 표 보기"):
        st.dataframe(
            gu_stat.sort_values("median", ascending=False, na_position="last")
            [["rank_median", "CGG_NM", "count", "median", "mean", "min", "max", "비고"]]
            .rename(columns={"rank_median": "순위", "CGG_NM": "자치구", "count": "거래건수", "median": "중위가(억)",
                              "mean": "평균가(억)", "min": "최소가(억)", "max": "최대가(억)"}),
            use_container_width=True, hide_index=True,
        )

# --------------------------------------------------------------------------------------
# 2) 건축년도별 가격
# --------------------------------------------------------------------------------------
with tab_year:
    st.subheader("건축년도별 물건금액 중위값 · 최대 · 최소")
    yr_df = work_df.dropna(subset=["ARCH_YR"]).copy()
    yr_df = yr_df[(yr_df["ARCH_YR"] >= 1960) & (yr_df["ARCH_YR"] <= CURRENT_YEAR)]
    yr_stat = (
        yr_df.groupby("ARCH_YR")["PRICE_EOK"]
        .agg(median="median", min="min", max="max", count="count")
        .reset_index()
        .sort_values("ARCH_YR")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yr_stat["ARCH_YR"], y=yr_stat["max"], line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=yr_stat["ARCH_YR"], y=yr_stat["min"], fill="tonexty",
                              fillcolor="rgba(255,140,0,0.15)", line=dict(width=0),
                              name="최소~최대 범위", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=yr_stat["ARCH_YR"], y=yr_stat["max"], mode="lines",
                              line=dict(color="rgba(255,99,71,0.6)", width=1), name="최대가"))
    fig.add_trace(go.Scatter(x=yr_stat["ARCH_YR"], y=yr_stat["min"], mode="lines",
                              line=dict(color="rgba(30,144,255,0.6)", width=1), name="최소가"))
    fig.add_trace(go.Bar(x=yr_stat["ARCH_YR"], y=yr_stat["median"], name="중위가격",
                          marker_color="rgba(220,80,20,0.85)"))
    fig.update_layout(
        height=520, xaxis_title="건축년도", yaxis_title="가격(억원)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(yr_stat, x="ARCH_YR", y="count", labels={"ARCH_YR": "건축년도", "count": "거래건수"},
                  title="건축년도별 거래건수")
    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("막대는 중위가격, 음영 영역은 해당 건축년도 거래의 최소~최대 가격 범위를 나타냅니다.")

# --------------------------------------------------------------------------------------
# 4) 건물용도별 가격
# --------------------------------------------------------------------------------------
with tab_usg:
    st.subheader("건물용도별 물건금액 분포 · 중위값 · 최대 · 최소")
    usg_stat = (
        work_df.groupby("BLDG_USG")["PRICE_EOK"]
        .agg(median="median", min="min", max="max", mean="mean", count="count")
        .reset_index()
        .sort_values("median", ascending=False)
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = px.box(work_df, x="BLDG_USG", y="PRICE_EOK", color="BLDG_USG", points=False,
                      labels={"BLDG_USG": "건물용도", "PRICE_EOK": "가격(억원)"})
        fig.update_layout(height=480, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.write("**건물용도별 요약 통계**")
        st.dataframe(
            usg_stat.rename(columns={"BLDG_USG": "건물용도", "median": "중위가(억)", "min": "최소가(억)",
                                      "max": "최대가(억)", "mean": "평균가(억)", "count": "거래건수"})
            .style.format({"중위가(억)": "{:.2f}", "최소가(억)": "{:.2f}", "최대가(억)": "{:.2f}", "평균가(억)": "{:.2f}"}),
            use_container_width=True, hide_index=True,
        )

    fig3 = px.pie(work_df, names="BLDG_USG", title="건물용도별 거래건수 비중", hole=0.45)
    fig3.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------------------------------------------
# 5) 층 - 가격 관계
# --------------------------------------------------------------------------------------
with tab_floor:
    st.subheader("층(FLR)과 가격의 관계")
    fl_df = work_df.dropna(subset=["FLR"]).copy()
    fl_df = fl_df[(fl_df["FLR"] >= -5) & (fl_df["FLR"] <= 80)]

    sample_df = fl_df.sample(min(len(fl_df), 4000), random_state=0) if len(fl_df) > 4000 else fl_df
    fig = px.scatter(sample_df, x="FLR", y="PRICE_EOK", color="BLDG_USG", opacity=0.5,
                      labels={"FLR": "층", "PRICE_EOK": "가격(억원)", "BLDG_USG": "건물용도"})
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    if len(fl_df) > 4000:
        st.caption(f"산점도는 표시 성능을 위해 전체 {len(fl_df):,}건 중 4,000건을 무작위 표본으로 표시했습니다.")

    bins = [-10, 0, 3, 6, 10, 15, 20, 30, 100]
    labels = ["지하", "1~3층", "4~6층", "7~10층", "11~15층", "16~20층", "21~30층", "31층 이상"]
    fl_df["FLR_BAND"] = pd.cut(fl_df["FLR"], bins=bins, labels=labels)
    band_stat = fl_df.groupby("FLR_BAND", observed=True)["PRICE_EOK"].agg(
        median="median", count="count").reset_index()

    fig2 = px.bar(band_stat, x="FLR_BAND", y="median", text="count",
                  labels={"FLR_BAND": "층 구간", "median": "중위가격(억원)"},
                  title="층 구간별 중위가격 (막대 위 숫자는 거래건수)")
    fig2.update_traces(texttemplate="%{text:,}건", textposition="outside")
    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------------------------------------
# 6) 연도별 특징 서술
# --------------------------------------------------------------------------------------
with tab_narrative:
    st.subheader("계약연도(CTRT_DAY 기준)별 거래 특징")
    ny_df = work_df.dropna(subset=["CTRT_YEAR"]).copy()
    ny_df["CTRT_YEAR"] = ny_df["CTRT_YEAR"].astype(int)

    year_summary = []
    for y, g in ny_df.groupby("CTRT_YEAR"):
        top_usg = g["BLDG_USG"].value_counts().idxmax() if not g["BLDG_USG"].empty else "-"
        top_gu = g["CGG_NM"].value_counts().idxmax() if not g["CGG_NM"].empty else "-"
        top_row = g.loc[g["PRICE_EOK"].idxmax()] if not g["PRICE_EOK"].empty else None
        year_summary.append({
            "연도": y, "거래건수": len(g), "중위가(억)": g["PRICE_EOK"].median(),
            "평균가(억)": g["PRICE_EOK"].mean(), "최고가(억)": g["PRICE_EOK"].max(),
            "최다용도": top_usg, "최다거래구": top_gu,
            "최고가거래": f"{top_row['CGG_NM']} {top_row['STDG_NM']} {top_row['BLDG_NM']}" if top_row is not None else "-",
        })
    year_summary_df = pd.DataFrame(year_summary).sort_values("연도")
    year_summary_df["전년대비 중위가 증감률(%)"] = year_summary_df["중위가(억)"].pct_change().mul(100).round(1)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=year_summary_df["연도"], y=year_summary_df["거래건수"], name="거래건수",
                          yaxis="y2", opacity=0.35, marker_color="lightsteelblue"))
    fig.add_trace(go.Scatter(x=year_summary_df["연도"], y=year_summary_df["중위가(억)"], name="중위가격(억원)",
                              mode="lines+markers", line=dict(color="firebrick", width=3)))
    fig.update_layout(
        height=450,
        yaxis=dict(title="중위가격(억원)"),
        yaxis2=dict(title="거래건수", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        year_summary_df.style.format({
            "중위가(억)": "{:.2f}", "평균가(억)": "{:.2f}", "최고가(억)": "{:.2f}",
            "전년대비 중위가 증감률(%)": "{:+.1f}%"
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("#### 연도별 요약")
    for _, r in year_summary_df.iterrows():
        change = r["전년대비 중위가 증감률(%)"]
        change_txt = "" if pd.isna(change) else f", 전년 대비 중위가격 {change:+.1f}%"
        st.markdown(
            f"- **{int(r['연도'])}년**: 총 {int(r['거래건수']):,}건 거래, 중위가격 {r['중위가(억)']:.2f}억원"
            f"{change_txt}. 가장 많이 거래된 유형은 **{r['최다용도']}**, "
            f"가장 활발했던 자치구는 **{r['최다거래구']}**. 최고가 거래는 {r['최고가거래']} ({r['최고가(억)']:.2f}억원)."
        )

# --------------------------------------------------------------------------------------
# 7) 원본 데이터
# --------------------------------------------------------------------------------------
with tab_raw:
    st.subheader("필터 적용된 원본 데이터")
    st.dataframe(work_df.drop(columns=["IS_CANCELLED"]), use_container_width=True, height=500)
    st.download_button(
        "CSV로 다운로드",
        data=work_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="seoul_realestate_filtered.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "본 대시보드는 서울 열린데이터광장 Open API(tbLnOpendataRtmsV)를 사용합니다. "
    "물건금액은 만원 단위로 제공되어 억원 단위로 환산했습니다. 자치구 경계는 southkorea/seoul-maps(KOSTAT 2013), "
    "법정동 좌표는 OpenStreetMap Nominatim을 사용했습니다."
)
