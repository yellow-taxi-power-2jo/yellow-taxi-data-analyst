# 광주_2반_김아영
"""
시각화Seaborn으로 정적 차트, Plotly로 인터랙티브 차트 각 1개 이상 작성(분포·상관관계·그룹
비교 중 택일)
"""

# ====================================================================
# 프로그램 전체 설명 및 변경 내역
# 작성자: 김아영 (시각화 담당)
# 작성목적:
# NYC Yellow Taxi 정제 데이터를 사용하여 Seaborn 정적 차트와
# Plotly 인터랙티브 차트를 생성하고 파일로 저장합니다.
# 작성일: 2026-07-21
# 프로그램명: End2End 데이터 분석 프로젝트 - 시각화 모듈
# 변경사항:
# - 2026-07-21, 최초 작성
# ====================================================================

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns


logger = logging.getLogger(__name__)

FIGURE_DIR = Path("outputs/figures")
HTML_DIR = Path("outputs/html")


def prepare_visualization_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    시각화에 필요한 컬럼을 생성하고 분석 가능한 범위로 데이터를 정리한다.

    생성 컬럼:
    - pickup_hour: 승차 시간대
    - trip_duration_min: 운행 시간(분)
    """
    logger.info("=" * 60)
    logger.info(" [4] 시각화용 데이터 준비")
    logger.info("=" * 60)

    required_columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "total_amount",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"시각화에 필요한 컬럼이 없습니다: {missing_columns}"
        )

    visual_df = df.copy()

    # 날짜·시간 컬럼의 타입을 안전하게 변환
    visual_df["tpep_pickup_datetime"] = pd.to_datetime(
        visual_df["tpep_pickup_datetime"],
        errors="coerce",
    )
    visual_df["tpep_dropoff_datetime"] = pd.to_datetime(
        visual_df["tpep_dropoff_datetime"],
        errors="coerce",
    )

    # 승차 시간에서 시간대 추출
    visual_df["pickup_hour"] = (
        visual_df["tpep_pickup_datetime"].dt.hour
    )

    # 운행 시간 계산
    visual_df["trip_duration_min"] = (
        visual_df["tpep_dropoff_datetime"]
        - visual_df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # 시각화 대상 컬럼의 결측치 제거
    visual_df = visual_df.dropna(
        subset=[
            "pickup_hour",
            "trip_distance",
            "total_amount",
            "trip_duration_min",
        ]
    )

    # 시각화를 왜곡할 수 있는 비정상 범위 제외
    visual_df = visual_df[
        (visual_df["trip_distance"] > 0)
        & (visual_df["trip_distance"] <= 50)
        & (visual_df["total_amount"] > 0)
        & (visual_df["total_amount"] <= 300)
        & (visual_df["trip_duration_min"] > 0)
        & (visual_df["trip_duration_min"] <= 180)
    ].copy()

    logger.info(
        "▶ 시각화용 데이터 준비 완료: %s행, %s열",
        f"{visual_df.shape[0]:,}",
        visual_df.shape[1],
    )

    return visual_df


def create_seaborn_hourly_chart(
    df: pd.DataFrame,
) -> Path:
    """
    Seaborn을 이용하여 시간대별 택시 운행 건수 막대그래프를 생성한다.

    시각화 유형:
    - 그룹 비교
    """
    logger.info("=" * 60)
    logger.info(" [5] Seaborn 정적 차트 생성")
    logger.info("=" * 60)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    hourly_counts = (
        df.groupby("pickup_hour")
        .size()
        .reset_index(name="trip_count")
        .sort_values("pickup_hour")
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=hourly_counts,
        x="pickup_hour",
        y="trip_count",
    )

    plt.title(
        "NYC Yellow Taxi Trips by Pickup Hour",
        fontsize=15,
        pad=15,
    )
    plt.xlabel("Pickup Hour")
    plt.ylabel("Number of Trips")
    plt.xticks(range(24))
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = FIGURE_DIR / "hourly_trip_count.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info(
        "▶ Seaborn 차트 저장 완료: %s",
        output_path,
    )

    return output_path


def create_plotly_distance_chart(
    df: pd.DataFrame,
    sample_size: int = 30_000,
) -> Path:
    """
    Plotly를 이용하여 이동 거리와 총 결제 금액의 관계를 나타내는
    인터랙티브 산점도를 생성한다.

    시각화 유형:
    - 상관관계
    """
    logger.info("=" * 60)
    logger.info(" [6] Plotly 인터랙티브 차트 생성")
    logger.info("=" * 60)

    HTML_DIR.mkdir(parents=True, exist_ok=True)

    # 전체 데이터를 모두 표현하면 브라우저가 느려질 수 있으므로 표본 추출
    if len(df) > sample_size:
        plot_df = df.sample(
            n=sample_size,
            random_state=42,
        )
    else:
        plot_df = df.copy()

    figure = px.scatter(
        plot_df,
        x="trip_distance",
        y="total_amount",
        color="pickup_hour",
        opacity=0.5,
        hover_data={
            "trip_distance": ":.2f",
            "total_amount": ":.2f",
            "pickup_hour": True,
            "trip_duration_min": ":.1f",
        },
        labels={
            "trip_distance": "Trip Distance (miles)",
            "total_amount": "Total Amount ($)",
            "pickup_hour": "Pickup Hour",
            "trip_duration_min": "Trip Duration (minutes)",
        },
        title=(
            "Relationship Between Trip Distance "
            "and Total Amount"
        ),
    )

    figure.update_layout(
        template="plotly_white",
        title_x=0.5,
    )

    output_path = (
        HTML_DIR / "distance_vs_total_amount.html"
    )

    figure.write_html(
        output_path,
        include_plotlyjs="cdn",
    )

    logger.info(
        "▶ Plotly 차트 저장 완료: %s",
        output_path,
    )
    logger.info(
        "▶ Plotly 시각화 표본 수: %s건",
        f"{len(plot_df):,}",
    )

    return output_path