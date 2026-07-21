#====================================================================
# 프로그램 전체 설명 및 변경 내역
# 작성자: 오영현 (통계 분석 담당), 정구현(통계분석 담당)
# 작성목적: 정제된 NYC Yellow Taxi 데이터의 기술통계(평균·표준편차·분위수)를
#           산출하고, 수치형 변수 간 상관계수를 계산합니다.
# 작성일: 2026-07-21
# 프로그램명: End2End 데이터 분석 프로젝트 - 통계 분석 모듈
# 변경사항 내역:
# - 2026-07-21, 최초 작성
# - 2026-07-21, 결제수단(신용카드 vs 현금)별 총요금 t-test(scipy.stats.ttest_ind) 및 p-value 해석 추가
#====================================================================

import logging

import pandas as pd
from scipy import stats

# 모듈 전용 로거 설정
logger = logging.getLogger(__name__)

# 기술통계·상관계수 계산 대상 수치형 컬럼.
# data/processed/cleaned_taxi_data.parquet(김용찬님 데이터 준비 파트 결과물)에
# 실제로 존재하는 컬럼명 기준으로 작성했다.
NUMERIC_COLUMNS = [
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "passenger_count",
]


def compute_descriptive_stats(df: pd.DataFrame, numeric_columns: list = None) -> pd.DataFrame:
    """
    [함수 설명] 수치형 컬럼의 평균·표준편차·분위수 등 기술통계를 산출합니다.

    지정한 컬럼이 DataFrame에 없으면 KeyError가 나기 전에 미리 확인해
    어떤 컬럼이 빠졌는지 명확한 오류 메시지를 낸다.
    """
    numeric_columns = numeric_columns or NUMERIC_COLUMNS

    logger.info("=" * 60)
    logger.info(" [7] 기술통계 산출 (평균·표준편차·분위수)")
    logger.info("=" * 60)

    missing_cols = set(numeric_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"기술통계 계산에 필요한 컬럼이 없습니다: {missing_cols}")
        raise ValueError(f"기술통계 계산에 필요한 컬럼이 없습니다: {missing_cols}")

    stats = df[numeric_columns].describe()
    logger.info(f"▶ 기술통계 결과:\n{stats}")

    return stats


def compute_correlation_matrix(df: pd.DataFrame, numeric_columns: list = None) -> pd.DataFrame:
    """
    [함수 설명] 수치형 변수 간 상관계수 행렬을 계산합니다.

    상관계수는 -1~1 사이 값으로, 1에 가까울수록 강한 양의 상관관계,
    -1에 가까울수록 강한 음의 상관관계, 0에 가까울수록 서로 무관하다는 뜻이다.
    """
    numeric_columns = numeric_columns or NUMERIC_COLUMNS

    logger.info("=" * 60)
    logger.info(" [8] 변수 간 상관계수 계산")
    logger.info("=" * 60)

    missing_cols = set(numeric_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"상관계수 계산에 필요한 컬럼이 없습니다: {missing_cols}")
        raise ValueError(f"상관계수 계산에 필요한 컬럼이 없습니다: {missing_cols}")

    corr = df[numeric_columns].corr()
    logger.info(f"▶ 상관계수 결과:\n{corr}")

    return corr


# t-test 관련 상수 (TLC 데이터 딕셔너리 기준 payment_type 코드)
#   1 = 신용카드(Credit card), 2 = 현금(Cash)
PAYMENT_CARD = 1
PAYMENT_CASH = 2

# 유의수준(귀무가설을 기각할지 판단하는 기준값)
ALPHA = 0.05


def run_payment_type_ttest(df: pd.DataFrame, value_column: str = "total_amount"):
    """
    [함수 설명] 결제수단(신용카드 vs 현금)에 따라 총요금(total_amount) 평균이
    통계적으로 유의하게 다른지 scipy.stats.ttest_ind로 독립표본 t-검정을 수행하고,
    p-value를 해석한다.

    - 귀무가설(H0): 두 결제수단의 평균 총요금은 같다.
    - 대립가설(H1): 두 결제수단의 평균 총요금은 다르다. (양측검정)
    - 두 그룹의 분산이 다를 수 있으므로 equal_var=False(Welch's t-test)를 사용한다.
    """
    logger.info("=" * 60)
    logger.info(" [9] 독립표본 t-검정 (결제수단별 총요금 차이)")
    logger.info("=" * 60)

    # 필요한 컬럼이 있는지 먼저 확인해 명확한 오류 메시지를 낸다.
    required_cols = {"payment_type", value_column}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.error(f"t-검정에 필요한 컬럼이 없습니다: {missing_cols}")
        raise ValueError(f"t-검정에 필요한 컬럼이 없습니다: {missing_cols}")

    # 두 그룹 추출 (결측치 제거)
    card = df.loc[df["payment_type"] == PAYMENT_CARD, value_column].dropna()
    cash = df.loc[df["payment_type"] == PAYMENT_CASH, value_column].dropna()

    logger.info(f"▶ 신용카드 그룹: n={len(card):,}, 평균={card.mean():.2f}, 표준편차={card.std():.2f}")
    logger.info(f"▶ 현금 그룹    : n={len(cash):,}, 평균={cash.mean():.2f}, 표준편차={cash.std():.2f}")

    # 각 그룹 표본이 최소 2개 이상이어야 t-검정이 가능하다.
    if len(card) < 2 or len(cash) < 2:
        logger.error("각 그룹의 표본 수가 부족하여 t-검정을 수행할 수 없습니다.")
        raise ValueError("t-검정을 위한 표본 수가 부족합니다.")

    # Welch's t-test (등분산 가정 없음)
    t_stat, p_value = stats.ttest_ind(card, cash, equal_var=False, nan_policy="omit")

    logger.info(f"▶ t-통계량(t-statistic): {t_stat:.4f}")
    logger.info(f"▶ p-value: {p_value:.6g}")

    # p-value 해석: 유의수준(ALPHA)과 비교해 귀무가설 기각 여부를 문장으로 남긴다.
    if p_value < ALPHA:
        higher = "신용카드" if card.mean() > cash.mean() else "현금"
        logger.info(f"💡 결론: p-value({p_value:.4g}) < α({ALPHA}) → 귀무가설 기각")
        logger.info("   두 결제수단의 평균 총요금은 '통계적으로 유의미한 차이'가 있다.")
        logger.info(f"   → 평균적으로 '{higher}' 결제 트립의 총요금이 더 높다.")
    else:
        logger.info(f"💡 결론: p-value({p_value:.4g}) ≥ α({ALPHA}) → 귀무가설 기각 실패")
        logger.info("   두 결제수단의 평균 총요금 차이는 '통계적으로 유의하지 않다'.")

    return t_stat, p_value