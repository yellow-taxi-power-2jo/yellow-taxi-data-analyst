"""
================================================================================
[프로젝트 명] 실습 4: 데이터 분석 및 ML 파이프라인 구축
[작성자] 캠퍼스명_반_이름
[설명] 
  - 1단계: EDA 시각화 4종 (2x2 서브플롯: 히스토그램, 박스플롯, 라인, 히트맵)
  - 2단계: 통계 검정 (t-test 및 카이제곱 검정 수행 및 해석)
  - 3단계: sklearn Pipeline을 이용한 전처리 및 모델 학습/저장
  - 4단계: Plotly를 이용한 인터랙티브 차트 생성 및 HTML 저장
================================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
import logging
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from pathlib import Path

# ---------------------------------------------------------
# [0단계] 로깅 설정 (Logging Setup)
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_practice_4():
    # 1. 데이터 로드
    try:
        # 질문자님이 지정한 실제 데이터 경로
        df = pd.read_parquet('data/raw/yellow_tripdata_2026-05.parquet')
        # 날짜 컬럼 변환 (라인 차트용)
        df['order_date'] = pd.to_datetime(df['tpep_pickup_datetime'])
        logger.info("✅ 데이터 로드 성공")
    except Exception as e:
        logger.error(f"❌ 데이터 로드 실패: {e}")
        return

    # ---------------------------------------------------------
    # [1단계] EDA 시각화 4종 (2×2 서브플롯)
    # ---------------------------------------------------------
    logger.info("[1단계] 2x2 시각화 생성 중...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # (1) 히스토그램: 요금 분포
    sns.histplot(df['total_amount'], kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('💰 요금 분포 (Histogram)')

    # (2) 박스플롯: VendorID별 요금 분포
    sns.boxplot(data=df, x='VendorID', y='total_amount', ax=axes[0, 1], palette='Set3')
    axes[0, 1].set_title('📍 VendorID별 요금 분포 (Boxplot)')

    # (3) 라인 차트: 월별 매출 추이
    monthly_sales = df.resample('ME', on='order_date')['total_amount'].sum() # 'M' 대신 'ME' 사용
    axes[1, 0].plot(monthly_sales.index, monthly_sales.values, marker='o', color='green')
    axes[1, 0].set_title('📅 월별 총 매출 추이 (Line Chart)')

    # (4) 히트맵: 상관관계
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1, 1])
    axes[1, 1].set_title('📊 변수 간 상관관계 (Heatmap)')

    plt.tight_layout()
    plt.show()
    logger.info("✅ 2x2 시각화 완료")

    # ---------------------------------------------------------
    # [2단계] 통계 검정 (t-test + 카이제곱)
    # ---------------------------------------------------------
    logger.info("[2단계] 통계 검정 수행 중...")

    # (1) t-test: VendorID(1 vs 2) 간의 요금 평균 차이 검정
    group_a = df[df['VendorID'] == 1]['total_amount']
    group_b = df[df['VendorID'] == 2]['total_amount']
    t_stat, p_val = stats.ttest_ind(group_a, group_b)

    logger.info(f"  - t-test 결과: t={t_stat:.3f}, p={p_val:.4f}")
    if p_val < 0.05:
        logger.info("    👉 해석: p < 0.05 이므로 두 그룹 간 요금 차이는 통계적으로 유의미합니다.")
    else:
        logger.info("    👉 해석: p >= 0.05 이므로 두 그룹 간 요금 차이는 통계적으로 유의미하지 않습니다.")

    # (2) 카이제곱 검정: VendorID와 payment_type 간의 독립성 검정
    contingency_table = pd.crosstab(df['VendorID'], df['payment_type'])
    chi2, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table)

    logger.info(f"  - 카이제곱 결과: chi2={chi2:.3f}, p={p_val_chi2:.4f}")
    if p_val_chi2 < 0.05:
        logger.info("    👉 해석: p < 0.05 이므로 Vendor와 결제 방식은 서로 연관성이 있습니다.")
    else:
        logger.info("    👉 해석: p >= 0.05 이므로 Vendor와 결제 방식은 서로 독립적입니다.")

    # ---------------------------------------------------------
    # [3단계] sklearn Pipeline 구성 및 저장
    # ---------------------------------------------------------
    logger.info("[3단계] ML Pipeline 구성 및 학습 중...")

    # 전처리할 컬럼 지정
    num_cols = ['total_amount']
    cat_cols = ['VendorID', 'payment_type']

    # 1. ColumnTransformer 설계
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(), cat_cols)
    ])

    # 2. Pipeline 구축
    model_pipeline = Pipeline([
        ('prep', preprocessor),
        ('reg', Ridge(alpha=1.0))
    ])

    # 3. 학습 및 평가
    X = df[cat_cols + num_cols]
    y = df['total_amount']
    model_pipeline.fit(X, y)
    
    score = model_pipeline.score(X, y)
    logger.info(f"✅ 모델 학습 완료. R2 Score: {score:.3f}")

    # 4. 모델 저장
    joblib.dump(model_pipeline, 'taxi_model_pipeline.pkl')
    logger.info("✅ 모델 파일 저장 완료: taxi_model_pipeline.pkl")

    # ---------------------------------------------------------
    # [4단계] Plotly 인터랙티브 차트 생성 및 저장
    # ---------------------------------------------------------
    logger.info("[4단계] Plotly 차트 생성 중...")

    # Plotly 막대 차트 생성
    fig_plotly = px.bar(
        df.sample(min(len(df), 5000)), 
        x='VendorID', 
        y='total_amount', 
        color='payment_type', 
        title='Vendor별 결제 방식에 따른 매출 현황 (Interactive)'
    )

    # HTML 파일로 저장
    fig_plotly.write_html('taxi_sales_interactive.html')
    logger.info("✅ Plotly 차트 저장 완료: taxi_sales_interactive.html")

    print("\n" + "="*50)
    print("🎉 [Practice 4] 모든 과제 완수 완료!")
    print("="*50)

if __name__ == "__main__":
    run_practice_4()
