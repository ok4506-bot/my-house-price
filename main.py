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
def build_url(api_key, start, end, rcpt_yr="", cgg_nm=""):
    """포지셔널 옵션 인자 중 앞부분(RCPT_YR, CGG_NM)만 사용, 뒤는 생략."""
    parts = [api_key, "json", SERVICE, str(start), str(end)]
    tail = [str(rcpt_yr), "", str(cgg_nm)]
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


def fetch_year(api_key, year, cgg_nm, max_pages, progress_cb=None):
    """특정 연도(+구) 데이터를 페이지네이션으로 모두 가져옴."""
    rows_all = []
    start = 1
    page = 0
    total_count = None
    while True:
        end = start + MAX_PER_CALL - 1
        url = build_url(api_key, start, end, rcpt_yr=year, cgg_nm=cgg_nm)
        try:
            resp = requests.get(url, timeout=25)
            data = resp.json()
        except Exception as e:
            return rows_all, total_count, f"요청 실패({year}년 {cgg_nm}): {e}"

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
            progress_cb(year, cgg_nm, page, len(rows_all), total_count)

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
def fetch_all(api_key, years, gu_list, max_pages_per_combo):
    """연도 x 자치구 조합으로 데이터를 모아 DataFrame으로 반환."""
    all_rows = []
    combos = [(y, g) for y in years for g in (gu_list if gu_list else [""])]
    errors = []
    prog = st.progress(0.0, text="데이터 수집 준비 중...")
    n = len(combos)
    for i, (y, g) in enumerate(combos):
        label = f"{y}년" + (f" · {g}" if g else " · 서울 전체")
        prog.progress((i) / max(n, 1), text=f"수집 중: {label} ({i+1}/{n})")
        rows, total, err = fetch_year(api_key, y, g, max_pages_per_combo)
        if err:
            errors.append(f"{label}: {err}")
        all_rows.extend(rows)
    prog.progress(1.0, text="수집 완료")
    time.sleep(0.2)
    prog.empty()
    return all_rows, errors


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


# --------------------------------------------------------------------------------------
# 지오코딩 (법정동 중심좌표) - Nominatim(OpenStreetMap) 사용, 결과 캐시
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=30 * 24 * 3600)
def geocode_dong(gu_nm, dong_nm):
    query = f"대한민국 서울특별시 {gu_nm} {dong_nm}"
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "countrycodes": "kr", "limit": 1}
    headers = {"User-Agent": "seoul-realestate-dashboard/1.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        js = r.json()
        if js:
            return float(js[0]["lat"]), float(js[0]["lon"])
    except Exception:
        pass
    return None, None


def geocode_many(pairs):
    """pairs: [(gu, dong), ...] 유니크 목록. 진행바 표시하며 순차 지오코딩(Nominatim 정책상 1건/초)."""
    results = {}
    prog = st.progress(0.0, text="법정동 좌표 조회 중 (최초 1회, 다소 시간이 걸립니다)")
    n = len(pairs)
    for i, (gu, dong) in enumerate(pairs):
        lat, lon = geocode_dong(gu, dong)
        results[(gu, dong)] = (lat, lon)
        prog.progress((i + 1) / max(n, 1), text=f"좌표 조회 중... {gu} {dong} ({i+1}/{n})")
        time.sleep(1.0)
    prog.empty()
    return results


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

all_gu_names = sorted([f["properties"]["name"] for f in load_gu_geojson()["features"]])
gu_selected = st.sidebar.multiselect("자치구 필터 (미선택 시 서울 전체)", options=all_gu_names, default=[])

max_pages = st.sidebar.number_input(
    "연도·자치구 조합당 최대 페이지 수 (1페이지=최대 1,000건)",
    min_value=1, max_value=500, value=10,
    help="호출 횟수를 제한합니다. 값이 클수록 더 많은 데이터를 가져오지만 API 호출 수와 대기 시간이 늘어납니다.",
)

st.sidebar.caption(
    "⚠️ 서울 열린데이터광장 API는 1회 요청당 최대 1,000건, 일반 인증키 기준 일일 호출 횟수 제한이 있습니다. "
    "넓은 기간을 조회할 때는 자치구를 함께 좁혀서 사용하는 것을 권장합니다."
)

if st.sidebar.button("🔄 새로고침 (캐시 지우고 다시 조회)", use_container_width=True):
    st.cache_data.clear()

year_list = [str(y) for y in range(years[0], years[1] + 1)]
with st.spinner("서울 열린데이터광장에서 실거래가 데이터를 불러오는 중..."):
    rows, fetch_errors = fetch_all(api_key, tuple(year_list), tuple(gu_selected), max_pages)
df = rows_to_df(rows)

if fetch_errors:
    with st.expander(f"⚠️ 일부 구간 수집 중 오류 발생 ({len(fetch_errors)}건) - 클릭해서 보기"):
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

tab_map_dong, tab_map_gu, tab_year, tab_usg, tab_floor, tab_narrative, tab_raw = st.tabs(
    ["🗺️ 법정동별 지도", "🗺️ 자치구별 지도", "🏗️ 건축년도별 가격", "🏢 건물용도별 가격",
     "🏬 층-가격 관계", "📅 연도별 특징", "📄 원본 데이터"]
)

# --------------------------------------------------------------------------------------
# 1) 법정동별 중위가격 지도
# --------------------------------------------------------------------------------------
with tab_map_dong:
    st.subheader("법정동(STDG_NM)별 물건금액 중위값 지도")
    st.caption("법정동 단위 좌표는 OpenStreetMap Nominatim으로 조회하며, 최초 조회 시 다소 시간이 걸릴 수 있습니다. "
               "표시 법정동 수를 제한하여 조회 시간을 줄일 수 있습니다.")
    top_n_dong = st.select_slider(
        "거래건수 상위 몇 개 법정동을 지도에 표시할까요?",
        options=[30, 75, 150, 225, 300],
        value=150,
        key="topn_dong",
    )
    show_dong_map = st.button("🗺️ 법정동별 중위가격 지도 보기", key="btn_dong_map")

    if show_dong_map:
        dong_stat = (
            work_df.groupby(["CGG_NM", "STDG_NM"])["PRICE_EOK"]
            .agg(median="median", count="count", mean="mean", max="max", min="min")
            .reset_index()
            .sort_values("count", ascending=False)
            .head(top_n_dong)
        )
        pairs = list(zip(dong_stat["CGG_NM"], dong_stat["STDG_NM"]))
        coords = geocode_many(pairs)
        dong_stat["lat"] = [coords.get((g, d), (None, None))[0] for g, d in pairs]
        dong_stat["lon"] = [coords.get((g, d), (None, None))[1] for g, d in pairs]
        dong_stat = dong_stat.dropna(subset=["lat", "lon"])

        if dong_stat.empty:
            st.warning("좌표를 조회하지 못했습니다. 잠시 후 다시 시도해주세요.")
        else:
            fig = px.scatter_mapbox(
                dong_stat, lat="lat", lon="lon",
                size="count", color="median",
                color_continuous_scale="YlOrRd",
                size_max=32, zoom=10,
                hover_name="STDG_NM",
                hover_data={"CGG_NM": True, "median": ":.2f", "mean": ":.2f",
                            "max": ":.2f", "min": ":.2f", "count": True, "lat": False, "lon": False},
                labels={"median": "중위가격(억원)"},
                center={"lat": 37.5546, "lon": 126.9706},
            )
            fig.update_layout(mapbox_style="carto-positron", height=650, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                dong_stat[["CGG_NM", "STDG_NM", "count", "median", "mean", "min", "max"]]
                .rename(columns={"CGG_NM": "자치구", "STDG_NM": "법정동", "count": "거래건수",
                                  "median": "중위가(억)", "mean": "평균가(억)", "min": "최소가(억)", "max": "최대가(억)"}),
                use_container_width=True, hide_index=True,
            )

# --------------------------------------------------------------------------------------
# 2) 자치구별 중위가격 지도
# --------------------------------------------------------------------------------------
with tab_map_gu:
    st.subheader("자치구(CGG_NM)별 물건금액 중위값 지도")
    show_gu_map = st.button("🗺️ 자치구별 중위가격 지도 보기", key="btn_gu_map")

    if show_gu_map:
        geojson = load_gu_geojson()
        all_gu_df = pd.DataFrame(
            [{"CGG_CD": f["properties"]["code"], "CGG_NM": f["properties"]["name"]} for f in geojson["features"]]
        )

        computed = (
            work_df.groupby(["CGG_CD", "CGG_NM"])["PRICE_EOK"]
            .agg(median="median", count="count", mean="mean", max="max", min="min")
            .reset_index()
        )
        # 서울시 25개 자치구 전체를 항상 표시 (데이터가 없는 구는 결측으로 표시)
        gu_stat = all_gu_df.merge(computed[["CGG_CD", "median", "count", "mean", "min", "max"]],
                                   on="CGG_CD", how="left")
        gu_stat["count"] = gu_stat["count"].fillna(0).astype(int)
        gu_stat["rank"] = gu_stat["median"].rank(ascending=False, method="min")

        n_missing = gu_stat["median"].isna().sum()
        if n_missing:
            st.caption(f"※ {n_missing}개 자치구는 현재 조건에서 거래 데이터가 없어 회색으로 표시됩니다.")

        fig = px.choropleth_mapbox(
            gu_stat, geojson=geojson, locations="CGG_CD", color="median",
            featureidkey="properties.code",
            color_continuous_scale="YlOrRd",
            mapbox_style="carto-positron",
            zoom=9.5, center={"lat": 37.5546, "lon": 126.9706},
            opacity=0.8,
            hover_name="CGG_NM",
            hover_data={"CGG_CD": False, "rank": True, "median": ":.2f", "mean": ":.2f",
                        "max": ":.2f", "min": ":.2f", "count": True},
            labels={"median": "중위가격(억원)", "rank": "가격순위"},
        )
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.update_layout(height=650, margin=dict(l=0, r=0, t=10, b=0),
                           coloraxis_colorbar=dict(title="중위가격(억원)"))
        st.plotly_chart(fig, use_container_width=True)

        ranked = gu_stat.dropna(subset=["median"]).sort_values("median", ascending=False)
        col1, col2 = st.columns(2)
        with col1:
            top5 = ranked.head(5)
            st.write("**중위가격 상위 5개 자치구**")
            st.dataframe(top5[["rank", "CGG_NM", "median", "count"]].rename(
                columns={"rank": "순위", "CGG_NM": "자치구", "median": "중위가(억)", "count": "거래건수"}),
                hide_index=True, use_container_width=True)
        with col2:
            bottom5 = ranked.tail(5).sort_values("median", ascending=True)
            st.write("**중위가격 하위 5개 자치구**")
            st.dataframe(bottom5[["rank", "CGG_NM", "median", "count"]].rename(
                columns={"rank": "순위", "CGG_NM": "자치구", "median": "중위가(억)", "count": "거래건수"}),
                hide_index=True, use_container_width=True)

        st.write("**서울시 25개 자치구 전체 순위**")
        st.dataframe(
            gu_stat.sort_values("median", ascending=False, na_position="last")
            [["rank", "CGG_NM", "count", "median", "mean", "min", "max"]]
            .rename(columns={"rank": "순위", "CGG_NM": "자치구", "count": "거래건수", "median": "중위가(억)",
                              "mean": "평균가(억)", "min": "최소가(억)", "max": "최대가(억)"}),
            use_container_width=True, hide_index=True,
        )

# --------------------------------------------------------------------------------------
# 3) 건축년도별 가격
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
