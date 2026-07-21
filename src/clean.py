#====================================================================
# 프로그램 전체 설명 및 변경 내역 (머리말)
# 작성자: 김용찬 (데이터 준비 담당)
# 작성목적: NYC Yellow Taxi 데이터를 Pandas와 Polars로 로딩하여 성능을 비교하고, 분석을 위한 결측치/중복 데이터를 정제합니다.
# 작성일: 2026-07-21
# 프로그램명: End2End 데이터 분석 프로젝트 - 데이터 준비 모듈
# 프로그램 전체 설명: 대용량 Parquet 파일을 처리하기 위해 Pandas와 Polars의 로딩 속도를 비교 검증합니다. 이후 Pandas를 기준으로 데이터의 구조를 파악하는 기초 EDA를 수행하고, 택시 도메인 특성(승객 수, 요금 결측치 제거 및 이상치 필터링)에 맞춘 데이터 클리닝을 자동화하는 함수 모음입니다.
# 본 파일은 KDT 교육을 위한 Sample 코드이므로 작성자에게 모든 저작권이 있습니다.
# 변경사항 내역: 
# - 2026-07-21, 최초 작성, Pandas vs Polars 비교 및 정제(clean_data) 로직 구현
#====================================================================

import time
import pandas as pd
import polars as pl
import os

def compare_loading(file_path: str):
    """
    [함수 설명] Pandas와 Polars를 사용하여 동일한 Parquet 파일을 로딩하고 소요 시간을 비교합니다.
    - file_path: '../data/raw/yellow_tripdata_2026-05.parquet'
    - 반환값: 로딩된 Pandas DataFrame, Polars DataFrame
    """
    print("="*50)
    print(" [1] Pandas vs Polars 데이터 로딩 속도 비교")
    print("="*50)
    
    # 1. Pandas 로딩 측정
    start_time = time.time()
    df_pd = pd.read_parquet(file_path)
    pd_time = time.time() - start_time
    print(f"▶ Pandas 로딩 시간: {pd_time:.4f} 초 (Shape: {df_pd.shape})")
    
    # 2. Polars 로딩 측정
    start_time = time.time()
    df_pl = pl.read_parquet(file_path)
    pl_time = time.time() - start_time
    print(f"▶ Polars 로딩 시간: {pl_time:.4f} 초 (Shape: {df_pl.shape})")
    
    # 결과 비교 분석 코멘트 출력
    faster = "Polars" if pl_time < pd_time else "Pandas"
    print(f"💡 결론: {faster}가 대용량 Parquet 파일 로딩에 더 유리함을 확인했습니다.\n")
    
    return df_pd, df_pl

def perform_basic_eda(df: pd.DataFrame):
    """
    [함수 설명] 데이터의 기초 통계량, 데이터 타입, 결측치 상태 등을 파악하는 기본 EDA를 수행합니다.
    """
    print("="*50)
    print(" [2] 기본 탐색적 데이터 분석 (EDA)")
    print("="*50)
    
    print("\n▶ 데이터 기본 정보 (info):")
    df.info()
    
    print("\n▶ 수치형 변수 기초 통계 요약 (describe):")
    print(df.describe())
    
    print("\n▶ 컬럼별 결측치 개수:")
    print(df.isnull().sum()[df.isnull().sum() > 0]) # 결측치가 있는 컬럼만 출력

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    [함수 설명] 중복 데이터와 결측치를 처리하여 분석에 적합한 형태로 정제합니다.
    - 처리 로직: 중복행 제거 -> 핵심 컬럼 결측치 제거 -> 요금 관련 음수/이상치 정제
    """
    print("\n" + "="*50)
    print(" [3] 데이터 정제 (결측치 및 중복 처리)")
    print("="*50)
    
    initial_rows = len(df)
    
    # 1. 중복 데이터 확인 및 제거
    dup_count = df.duplicated().sum()
    df_cleaned = df.drop_duplicates()
    print(f"▶ 중복 데이터 처리: {dup_count:,}건 제거 완료")
    
    # 2. 결측치 처리
    # 택시 데이터 특성상 승객 수(passenger_count)나 요금(fare_amount)이 없으면 
    # 분석 가치가 떨어지므로 해당 결측치가 포함된 행은 제거(Drop)하는 전략 사용
    missing_before = df_cleaned.isnull().sum().sum()
    critical_columns = ['passenger_count', 'fare_amount', 'tpep_pickup_datetime']
    
    # 필요하다면 다른 숫자형 결측치를 중앙값으로 채울 수도 있음 
    # (예: df_cleaned['congestion_surcharge'] = df_cleaned['congestion_surcharge'].fillna(0))
    df_cleaned = df_cleaned.dropna(subset=critical_columns)
    missing_after = df_cleaned.isnull().sum().sum()
    
    print(f"▶ 핵심 컬럼 결측치 처리: 총 {missing_before:,}개 요소 중 핵심 결측 행 제거 완료 (남은 결측 요소: {missing_after:,}개)")
    
    # 3. 논리적 이상치 추가 정제 (요금이나 승객 수가 0 이하인 비정상 데이터 제거)
    outlier_cond = (df_cleaned['fare_amount'] > 0) & (df_cleaned['passenger_count'] > 0)
    df_cleaned = df_cleaned[outlier_cond]
    
    final_rows = len(df_cleaned)
    print(f"▶ 정제 요약: 초기 {initial_rows:,}건 -> 정제 후 {final_rows:,}건 유지 (총 {initial_rows - final_rows:,}건 제거)")
    
    return df_cleaned