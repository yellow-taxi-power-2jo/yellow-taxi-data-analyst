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
import random

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

# [표본 크기별 p-value 실험] 설정
#   같은 두 집단(신용카드 vs 현금)에서 표본 크기(n)만 10 → 100 → 1000 …으로 늘려가며
#   p-value가 어떻게 변하는지 관찰하기 위한 값들이다.
#   '평균 차이(효과크기)'는 그대로인데도 n이 커지면 p-value가 급격히 작아지는 것을 보여준다.
SAMPLE_SIZES = [10, 100, 1000, 10000, 100000]
# 각 표본 크기마다 무작위 추출을 몇 번 반복해 평균을 낼지 (우연에 덜 흔들리도록).
N_REPEATS = 10
# 무작위 추출 재현성을 위한 시드
RANDOM_SEED = 42


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
    logger.info(f"▶ p-value: {p_value:.6f}")

    # p-value 해석: 유의수준(ALPHA)과 비교해 귀무가설 기각 여부를 문장으로 남긴다.
    if p_value < ALPHA:
        higher = "신용카드" if card.mean() > cash.mean() else "현금"
        logger.info(f"💡 결론: p-value({p_value:.6f}) < α({ALPHA}) → 귀무가설 기각")
        logger.info("   두 결제수단의 평균 총요금은 '통계적으로 유의미한 차이'가 있다.")
        logger.info(f"   → 평균적으로 '{higher}' 결제 트립의 총요금이 더 높다.")
    else:
        logger.info(f"💡 결론: p-value({p_value:.6f}) ≥ α({ALPHA}) → 귀무가설 기각 실패")
        logger.info("   두 결제수단의 평균 총요금 차이는 '통계적으로 유의하지 않다'.")

    # p-value 해석의 연장선: 표본 크기(n)를 늘려가며 p-value가 어떻게 변하는지 실험한다.
    run_sample_size_experiment(card, cash)

    return t_stat, p_value


def run_sample_size_experiment(card: pd.Series, cash: pd.Series):
    """
    [함수 설명] 같은 두 집단에서 표본 크기(n)만 10 → 100 → 1000 …으로 늘려가며
    평균 p-value가 어떻게 변하는지 보여주는 실험이다.

    핵심 메시지: 두 집단의 '실제 평균 차이(효과크기)'는 변하지 않는데도,
    표본이 커질수록 p-value는 급격히 작아진다. 즉 전체 데이터(수백만 건)에서
    p-value가 0.0000으로 나오는 것은 버그가 아니라 '큰 표본의 자연스러운 결과'다.

    각 표본 크기마다 무작위 추출을 N_REPEATS번 반복해 p-value의 평균을 내고,
    p < ALPHA(유의)로 나온 비율도 함께 보여준다.
    """
    logger.info("=" * 60)
    logger.info(" [9-1] 표본 크기별 p-value 변화 실험")
    logger.info("=" * 60)
    logger.info("   같은 두 집단에서 표본 크기(n)만 늘려가며 p-value를 관찰한다.")
    logger.info("   (평균 차이는 그대로여도 n이 커지면 p-value가 작아지는지 확인)")

    rng = random.Random(RANDOM_SEED)
    # 표頭
    logger.info(f"   {'표본크기 n':>10} | {'평균 p-value':>14} | {'유의(p<0.05) 비율':>16} | {'평균 t':>10}")
    logger.info(f"   {'-'*10} | {'-'*14} | {'-'*16} | {'-'*10}")

    for n in SAMPLE_SIZES:
        # 두 그룹 중 더 작은 쪽보다 큰 표본은 뽑을 수 없으므로 건너뛴다.
        if n > len(card) or n > len(cash):
            logger.info(f"   {n:>10,} | (표본 수 부족으로 건너뜀)")
            continue

        p_values = []
        t_stats = []
        n_significant = 0
        for _ in range(N_REPEATS):
            # 각 그룹에서 서로 다른 시드로 n개씩 무작위 추출
            card_sample = card.sample(n=n, random_state=rng.randint(0, 2**31 - 1))
            cash_sample = cash.sample(n=n, random_state=rng.randint(0, 2**31 - 1))
            t_stat, p_value = stats.ttest_ind(card_sample, cash_sample, equal_var=False)
            p_values.append(p_value)
            t_stats.append(t_stat)
            if p_value < ALPHA:
                n_significant += 1

        avg_p = sum(p_values) / len(p_values)
        avg_t = sum(t_stats) / len(t_stats)
        sig_ratio = n_significant / N_REPEATS
        logger.info(f"   {n:>10,} | {avg_p:>14.6f} | {sig_ratio:>15.0%} | {avg_t:>10.2f}")

    logger.info("💡 해석: 두 집단의 실제 평균 차이는 그대로인데도, 표본 n이 커질수록")
    logger.info("   p-value는 0에 수렴하고 '유의' 판정 비율은 100%에 가까워진다.")
    logger.info("   → 전체 데이터의 p-value=0.0000은 오류가 아니라 '큰 표본'의 당연한 결과다.")