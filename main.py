#====================================================================
# 작성자: 김용찬, 김아영, 오영현, 정구현, 이효빈, 김세빈
# 작성목적: 데이터 준비 + 시각화 + 통계 분석 + ML Pipeline + report.md
#           자동생성까지 전체 파이프라인을 실행하고 로깅하는 스크립트
# 변경사항 내역:
# - 2026-07-21, 결제수단별 총요금 t-test 파이프라인 단계([9]) 연결
# - 2026-07-21, report.md 자동 생성 단계([11])를 main.py에 연결
#               (compare_loading의 로딩 시간, clean_data 전후 행 수,
#                기술통계·상관계수·t-test·ML Pipeline 결과를 전부 받아
#                report_generator.generate_report_md()로 전달)
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
from src.ml_pipeline import train_evaluate_and_save_model
from src.analysis import (
    compute_descriptive_stats,
    compute_correlation_matrix,
    run_payment_type_ttest,
)
from src.report_generator import generate_report_md

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
MODEL_PATH = os.path.join('models', 'taxi_classifier_pipeline.joblib')

def main():
    logger = setup_logger()
    logger.info("🚀 [End2End 프로젝트] 데이터 준비 파이프라인을 시작합니다.\n")
    
    try:
        # 1. Pandas vs Polars 속도 비교 로딩
        df_pd, _, pandas_time, polars_time = compare_loading(RAW_DATA_PATH)
        initial_rows = len(df_pd)

        # 2. 기초 EDA 수행
        perform_basic_eda(df_pd)
        
        # 3. 데이터 정제 (결측치/중복/이상치 처리)
        df_cleaned = clean_data(df_pd)
        final_rows = len(df_cleaned)
        
        # 4. 정제 완료 데이터 저장 (data/processed/)
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        df_cleaned.to_parquet(PROCESSED_DATA_PATH, index=False)


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
        stats_result = compute_descriptive_stats(df_cleaned)
        corr_result = compute_correlation_matrix(df_cleaned)

        # 9. 결제수단별 총요금 t-검정 및 p-value 해석
        ttest_result = run_payment_type_ttest(df_cleaned)

        # 10. sklearn Pipeline 기반 전처리 + 모델 학습 + 평가 + 저장
        pipeline_metrics = train_evaluate_and_save_model(df_cleaned, MODEL_PATH)
        
        logger.info(f"\n✅ 데이터 준비 완료! 정제된 파일 저장 위치: '{PROCESSED_DATA_PATH}'")
        logger.info(f"✅ 학습 모델 저장 위치: '{MODEL_PATH}'")

        # 11. report.md 자동 생성 (Jinja2)
        generate_report_md(
            pandas_time=pandas_time,
            polars_time=polars_time,
            initial_rows=initial_rows,
            final_rows=final_rows,
            seaborn_chart_path=str(seaborn_output_path),
            plotly_chart_path=str(plotly_output_path),
            descriptive_stats=stats_result,
            correlation_matrix=corr_result,
            ttest_result=ttest_result,
            pipeline_metrics=pipeline_metrics,
            model_path=MODEL_PATH,
        )

    except Exception as e:
        logger.error(f"❌ [파이프라인 오류 발생]: {e}", exc_info=True)

if __name__ == "__main__":
    main()