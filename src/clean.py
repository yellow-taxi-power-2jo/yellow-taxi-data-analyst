#====================================================================
# 프로그램 전체 설명 및 변경 내역 (머리말)
# 작성자: 김용찬 (데이터 준비 담당)
# 작성목적: NYC Yellow Taxi 데이터를 Pandas와 Polars로 로딩하여 성능을 비교하고, 결측치/중복 데이터를 정제합니다.
# 작성일: 2026-07-21
# 프로그램명: End2End 데이터 분석 프로젝트 - 데이터 준비 모듈
# 변경사항 내역: 
# - 2026-07-21, 최초 작성
# - 2026-07-21, print 출력 방식을 파이썬 표준 logging 시스템으로 변경
#====================================================================

import time
import logging
import os
import pandas as pd
import polars as pl

# 모듈 전용 로거 설정
logger = logging.getLogger(__name__)

def compare_loading(file_path: str):
    """
    [함수 설명] Pandas와 Polars를 사용하여 동일한 Parquet 파일을 로딩하고 소요 시간을 비교합니다.
    """
    logger.info("="*60)
    logger.info(" [1] Pandas vs Polars 데이터 로딩 속도 비교")
    logger.info("="*60)
    
    if not os.path.exists(file_path):
        logger.error(f"원본 데이터 파일이 경로에 존재하지 않습니다: {file_path}")
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    # 1. Pandas 로딩 측정
    start_time = time.time()
    df_pd = pd.read_parquet(file_path)
    pd_time = time.time() - start_time
    logger.info(f"▶ Pandas 로딩 시간: {pd_time:.4f} 초 | Shape: {df_pd.shape}")
    
    # 2. Polars 로딩 측정
    start_time = time.time()
    df_pl = pl.read_parquet(file_path)
    pl_time = time.time() - start_time
    logger.info(f"▶ Polars 로딩 시간: {pl_time:.4f} 초 | Shape: {df_pl.shape}")
    
    # 결과 비교 분석
    faster = "Polars" if pl_time < pd_time else "Pandas"
    diff_ratio = abs(pd_time - pl_time) / max(pd_time, pl_time) * 100
    logger.info(f"💡 결론: {faster}가 약 {diff_ratio:.1f}% 더 빠르게 대용량 데이터를 처리했습니다.\n")
    
    return df_pd, df_pl

def perform_basic_eda(df: pd.DataFrame):
    """
    [함수 설명] 데이터의 데이터 타입, 요약 통계량, 결측치 상태 등 기본 EDA를 수행합니다.
    """
    logger.info("="*60)
    logger.info(" [2] 기본 탐색적 데이터 분석 (EDA)")
    logger.info("="*60)
    
    logger.info(f"▶ 데이터 행/열 개수: {df.shape[0]:,}행, {df.shape[1]}열")
    logger.info(f"▶ 전체 컬럼 목록: {list(df.columns)}")
    
    # 결측치 상태 로깅
    null_series = df.isnull().sum()
    missing_cols = null_series[null_series > 0]
    if len(missing_cols) > 0:
        logger.info(f"▶ 결측치가 존재하는 컬럼 현황:\n{missing_cols.to_string()}")
    else:
        logger.info("▶ 결측치가 있는 컬럼이 없습니다.")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    [함수 설명] 중복 데이터 및 결측치, 비논리적 이상치를 정제합니다.
    """
    logger.info("\n" + "="*60)
    logger.info(" [3] 데이터 정제 (결측치, 중복 및 이상치 처리)")
    logger.info("="*60)
    
    initial_rows = len(df)
    
    # 1. 중복 데이터 제거
    dup_count = df.duplicated().sum()
    df_cleaned = df.drop_duplicates()
    logger.info(f"▶ 중복 데이터 제거: 총 {dup_count:,}건 삭제")
    
    # 2. 주요 컬럼 결측치 제거
    missing_before = df_cleaned.isnull().sum().sum()
    critical_columns = ['passenger_count', 'fare_amount', 'tpep_pickup_datetime']
    df_cleaned = df_cleaned.dropna(subset=critical_columns)
    missing_after = df_cleaned.isnull().sum().sum()
    logger.info(f"▶ 핵심 결측치 처리: 전체 결측 요소 {missing_before:,}개 중 주요 결측 행 제거 (남은 결측 요소: {missing_after:,}개)")
    
    # 3. 비논리 이상치 정제 (택시 요금 및 승객 수가 0 이하인 데이터 제거)
    outlier_cond = (df_cleaned['fare_amount'] > 0) & (df_cleaned['passenger_count'] > 0)
    df_cleaned = df_cleaned[outlier_cond]
    
    final_rows = len(df_cleaned)
    logger.info(f"▶ 최종 정제 완료: {initial_rows:,}건 -> {final_rows:,}건 (총 {initial_rows - final_rows:,}건 정제됨)")
    
    return df_cleaned