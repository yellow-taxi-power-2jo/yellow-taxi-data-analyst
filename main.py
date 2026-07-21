#====================================================================
# 작성자: 김용찬, 김아영, 오영현
# 작성목적: 데이터 준비 + 시각화 + 통계 분석 파이프라인 자동 실행 및 로깅 스크립트
# Seaborn 정적 차트, Plotly 인터랙티브 차트 생성
# 변경사항 내역:
# - 2026-07-21, 결제수단별 총요금 t-test 파이프라인 단계([9]) 연결
#====================================================================

import os
import logging
from datetime import datetime
from src.clean import compare_loading, perform_basic_eda, clean_data
from src.visualization import (
    create_plotly_distance_chart,
    create_seaborn_hourly_chart,
    prepare_visualization_data,
)
from src.analysis import (
    compute_descriptive_stats,
    compute_correlation_matrix,
    run_payment_type_ttest,
)

def setup_logger():
    """
    콘솔(터미널)과 파일 양쪽으로 로그를 남기도록 설정합니다.
    - 로그 파일 생성 위치: logs/data_prep_YYYYMMDD.log
    """
    os.makedirs('logs', exist_ok=True)
    log_filename = datetime.now().strftime('logs/data_prep_%Y%m%d.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),  # 파일 저장
            logging.StreamHandler()                                # 콘솔 화면 출력
        ]
    )
    return logging.getLogger(__name__)

# 파일 경로 설정 (상대 경로)
RAW_DATA_PATH = os.path.join('data', 'raw', 'yellow_tripdata_2026-05.parquet')
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'cleaned_taxi_data.parquet')

def main():
    logger = setup_logger()
    logger.info("🚀 [End2End 프로젝트] 데이터 준비 파이프라인을 시작합니다.\n")
    
    try:
        # 1. Pandas vs Polars 속도 비교 로딩
        df_pd, _ = compare_loading(RAW_DATA_PATH)
        
        # 2. 기초 EDA 수행
        perform_basic_eda(df_pd)
        
        # 3. 데이터 정제 (결측치/중복/이상치 처리)
        df_cleaned = clean_data(df_pd)
        
        # 4. 정제 완료 데이터 저장 (data/processed/)
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        df_cleaned.to_parquet(PROCESSED_DATA_PATH, index=False)
        
        logger.info(f"\n✅ 데이터 준비 완료! 정제된 파일 저장 위치: '{PROCESSED_DATA_PATH}'")

        # 5. 시각화용 데이터 준비
        visualization_df = prepare_visualization_data(df_cleaned)

        # 6. Seaborn 정적 차트 생성
        seaborn_output_path = create_seaborn_hourly_chart(
            visualization_df
        )

        # 7. Plotly 인터랙티브 차트 생성
        plotly_output_path = create_plotly_distance_chart(
            visualization_df
        )

        logger.info(
            f"✅ Seaborn 정적 차트 저장 위치: '{seaborn_output_path}'"
        )
        logger.info(
            f"✅ Plotly 인터랙티브 차트 저장 위치: '{plotly_output_path}'"
        )

        # 8. 기술통계 및 상관계수 계산
        compute_descriptive_stats(df_cleaned)
        compute_correlation_matrix(df_cleaned)

        # 9. 결제수단별 총요금 t-검정 및 p-value 해석
        run_payment_type_ttest(df_cleaned)

    except Exception as e:
        logger.error(f"❌ [파이프라인 오류 발생]: {e}", exc_info=True)

if __name__ == "__main__":
    main()