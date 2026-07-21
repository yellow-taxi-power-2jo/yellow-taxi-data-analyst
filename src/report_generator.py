#====================================================================
# 프로그램 전체 설명 및 변경 내역 (머리말)
# 작성자: 김세빈
# 작성목적: 팀원별 분석 결과(데이터준비/시각화/통계분석/t-test/ML Pipeline)를
#           Jinja2 템플릿(templates/report_template.md.j2)에 채워 넣어
#           report.md를 자동 생성합니다.
# 작성일: 2026-07-21
# 프로그램명: End2End 데이터 분석 프로젝트 - 자동화 리포트 모듈
# 변경사항 내역:
# - 2026-07-21, 최초 작성 (Jinja2 템플릿 방식)
# - 2026-07-21, ml_pipeline.train_evaluate_and_save_model()이 반환하는
#               실제 metrics dict(accuracy/precision/recall/f1/roc_auc)와
#               analysis.run_payment_type_ttest()가 반환하는 dict(t_stat/
#               p_value/interpretation 등)에 맞춰 템플릿 변수 이름을 정리함
#====================================================================

import logging
import os
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
REPORT_PATH = os.path.join(BASE_DIR, "report.md")


def generate_report_md(
    pandas_time: float,
    polars_time: float,
    initial_rows: int,
    final_rows: int,
    seaborn_chart_path: str,
    plotly_chart_path: str,
    descriptive_stats,
    correlation_matrix,
    ttest_result: dict,
    pipeline_metrics: dict,
    model_path: str,
    model_name: str = "SGDClassifier Pipeline",
    report_path: str = REPORT_PATH,
) -> None:
    """
    [함수 설명] 팀원별 분석 결과를 Jinja2 템플릿에 채워 넣어 report.md 파일로
    저장합니다.

    - ttest_result: analysis.run_payment_type_ttest()의 반환값(dict).
      t_stat/p_value/interpretation 키를 사용한다.
    - pipeline_metrics: ml_pipeline.train_evaluate_and_save_model()의
      반환값(dict). accuracy/f1/roc_auc 키를 사용한다.

    템플릿 파일이 없으면 FileNotFoundError를 명시적으로 발생시켜 원인을
    바로 알 수 있게 한다.
    """
    logger.info("=" * 60)
    logger.info(" [11] report.md 자동 생성 (Jinja2)")
    logger.info("=" * 60)

    template_file = os.path.join(TEMPLATE_DIR, "report_template.md.j2")
    if not os.path.exists(template_file):
        logger.error(f"템플릿 파일을 찾을 수 없습니다: {template_file}")
        raise FileNotFoundError(f"템플릿 파일을 찾을 수 없습니다: {template_file}")

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template("report_template.md.j2")

    rendered = template.render(
        generated_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        pandas_time=f"{pandas_time:.4f}",
        polars_time=f"{polars_time:.4f}",
        initial_rows=f"{initial_rows:,}",
        final_rows=f"{final_rows:,}",
        removed_rows=f"{initial_rows - final_rows:,}",
        seaborn_chart_path=seaborn_chart_path,
        plotly_chart_path=plotly_chart_path,
        descriptive_stats=descriptive_stats,
        correlation_matrix=correlation_matrix,
        t_stat=f"{ttest_result['t_stat']:.4f}",
        p_val=f"{ttest_result['p_value']:.6g}",
        t_test_interpretation=ttest_result["interpretation"],
        model_name=model_name,
        model_accuracy=f"{pipeline_metrics.get('accuracy', float('nan')):.4f}",
        model_f1=f"{pipeline_metrics.get('f1', float('nan')):.4f}",
        model_roc_auc=(
            f"{pipeline_metrics['roc_auc']:.4f}"
            if "roc_auc" in pipeline_metrics else "N/A"
        ),
        model_path=model_path,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(rendered)

    logger.info(f"▶ report.md 생성 완료: '{report_path}'")