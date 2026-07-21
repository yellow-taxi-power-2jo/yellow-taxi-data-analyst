#====================================================================
# 프로그램 전체 설명 및 변경 내역
# 작성자: 오영현 (통계 분석 담당)
# 작성목적: 정제된 NYC Yellow Taxi 데이터의 기술통계(평균·표준편차·분위수)를
#           산출하고, 수치형 변수 간 상관계수를 계산합니다.
# 작성일: 2026-07-21
# 프로그램명: End2End 데이터 분석 프로젝트 - 통계 분석 모듈
# 변경사항 내역:
# - 2026-07-21, 최초 작성
#====================================================================

import logging

import pandas as pd

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