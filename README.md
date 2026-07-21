# 🚕 NYC Yellow Taxi 데이터 분석 팀 프로젝트

## 프로젝트 목적

NYC Yellow Taxi(2026년 5월) 운행 데이터를 pandas·Polars 양쪽으로 로딩해 비교하고,
결측치·중복을 정제한 뒤 시각화·통계분석·머신러닝 Pipeline까지 수행해
결제수단(신용카드/현금)에 따른 요금 차이를 검정하고, 결제수단을 예측하는
분류 모델을 만드는 End-to-End 데이터 분석 프로젝트입니다.

## 프로젝트 구조

```
yellow-taxi-data-analyst/
├── data/
│   ├── raw/                      # 원본 데이터 (yellow_tripdata_2026-05.parquet)
│   └── processed/                # 정제 완료 데이터 (cleaned_taxi_data.parquet)
├── src/
│   ├── clean.py                  # 데이터 로딩(Pandas/Polars 비교) + 정제 + EDA
│   ├── visualization.py          # Seaborn 정적 차트 + Plotly 인터랙티브 차트
│   ├── analysis.py               # 기술통계·상관계수 + 결제수단별 t-test
│   ├── ml_pipeline.py            # sklearn Pipeline 전처리+모델 학습·평가·저장
│   └── report_generator.py       # Jinja2로 report.md 자동 생성
├── templates/
│   └── report_template.md.j2     # report.md Jinja2 템플릿
├── outputs/
│   ├── figures/                  # Seaborn 정적 차트(PNG)
│   └── html/                     # Plotly 인터랙티브 차트(HTML)
├── models/
│   └── taxi_classifier_pipeline.joblib   # 학습된 모델
├── logs/                         # 실행 로그 (data_prep_YYYYMMDD.log)
├── main.py                       # 전체 파이프라인 실행 스크립트
├── report.md                     # 자동 생성된 분석 결과 리포트
├── requirements.txt
└── README.md
```

## 환경 설정

```bash
# 가상환경 생성 및 활성화 (uv 기준)
uv venv
source .venv/bin/activate      # macOS/Linux

# 패키지 설치
uv pip install -r requirements.txt
```

## 데이터 준비

[NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)에서
2026년 5월 Yellow Taxi 데이터를 다운로드해 `data/raw/`에 저장하세요.

```bash
mkdir -p data/raw
curl -o data/raw/yellow_tripdata_2026-05.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet
```

## 실행 방법

```bash
python main.py
```

실행하면 다음이 순서대로 진행됩니다.

1. Pandas·Polars 로딩 속도 비교
2. 기초 EDA (결측치 확인 등)
3. 데이터 정제 (결측치·중복·이상치 처리) → `data/processed/cleaned_taxi_data.parquet` 저장
4. Seaborn 정적 차트 생성 → `outputs/figures/`
5. Plotly 인터랙티브 차트 생성 → `outputs/html/`
6. 기술통계·상관계수 산출
7. 결제수단(신용카드 vs 현금)별 총요금 t-test + p-value 해석
8. sklearn Pipeline(전처리+모델) 학습·평가 → `models/taxi_classifier_pipeline.joblib` 저장
9. 분석 결과를 `report.md`로 자동 생성

## 결과물 확인

```bash
cat report.md                                    # 전체 분석 결과 리포트
open outputs/figures/hourly_trip_count.png       # Seaborn 차트
open outputs/html/distance_vs_total_amount.html  # Plotly 인터랙티브 차트
```

## 코드 스타일 검사

```bash
ruff check .
```

## Git 워크플로

```bash
git add .
git commit -m "설명 메시지"
git push origin main
```

## 팀원 및 역할 분담

| 이름 | 담당 파트 |
|---|---|
| 김용찬 | 데이터 준비 (Pandas·Polars 로딩 비교, 결측치·중복 처리, EDA) |
| 김아영 | 시각화 (Seaborn 정적 차트, Plotly 인터랙티브 차트) |
| 오영현 | 통계 분석 - 기술통계·상관계수 |
| 정구현 | 통계 분석 - t-test·p-value 해석 |
| 이효빈 | ML Pipeline (전처리+모델 학습, 평가지표, 모델 저장) |
| 김세빈 | 자동화 - report.md 자동 생성 |