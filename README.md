# 서울시 부동산 실거래가 대시보드

서울 열린데이터광장 Open API(`tbLnOpendataRtmsV`)를 이용해 서울시 부동산 실거래가를
법정동/자치구 지도, 건축년도별·건물용도별·층별 가격 분석, 연도별 거래 특징 요약으로
보여주는 Streamlit 앱입니다.

## 파일 구성
- `main.py` : Streamlit 앱 본체
- `requirements.txt` : 의존 패키지 목록
- `seoul_gu.geojson` : 서울시 자치구 경계 (southkorea/seoul-maps, KOSTAT 2013 기반)

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run main.py
```

## 인증키 준비
1. https://data.seoul.go.kr 에서 회원가입 후 Open API 인증키를 발급받으세요.
2. 앱 실행 후 왼쪽 사이드바에 인증키를 직접 입력하거나,
   `.streamlit/secrets.toml` 파일에 아래처럼 넣어두면 자동으로 채워집니다.

```toml
# .streamlit/secrets.toml
SEOUL_API_KEY = "발급받은_인증키"
```

인증키가 없어도 사이드바의 "인증키 없이 샘플 데이터로 체험하기"를 선택하면
샘플 데이터로 대시보드 기능을 먼저 확인할 수 있습니다.

## Streamlit Cloud 배포
1. 이 폴더(`main.py`, `requirements.txt`, `seoul_gu.geojson`)를 GitHub 저장소에 올립니다.
2. https://share.streamlit.io 에서 새 앱을 생성하고 저장소/브랜치/`main.py`를 지정합니다.
3. 앱 설정(Settings → Secrets)에 `SEOUL_API_KEY = "발급받은_인증키"`를 추가합니다.

## 참고 사항
- 서울 열린데이터광장 API는 1회 요청당 최대 1,000건만 반환하며, 일반 인증키는
  일일 호출 횟수 제한이 있습니다. 조회 기간·자치구 범위가 넓을수록 호출 횟수와
  대기 시간이 늘어나므로, 사이드바의 "연도·자치구 조합당 최대 페이지 수"로 조절하세요.
- 법정동 지도의 좌표는 OpenStreetMap Nominatim으로 실시간 조회하며, 결과는
  앱 세션 동안 캐시됩니다(최초 조회 시에만 시간이 걸립니다).
- 물건금액(THING_AMT)은 만원 단위로 제공되어 앱에서는 억원 단위로 환산해 표시합니다.
- 취소된 거래(RTRCN_DAY 값이 있는 행)는 기본적으로 분석에서 제외되며, 사이드바에서
  다시 포함하도록 설정할 수 있습니다.
