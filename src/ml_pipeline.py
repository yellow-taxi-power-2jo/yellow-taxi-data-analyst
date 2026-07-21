#====================================================================
# 프로그램 전체 설명 및 변경 내역 (머리말)
# 작성자: 이효빈
# 작성목적: 정제된 NYC Yellow Taxi 데이터로 전처리+모델을 하나의 sklearn
#           Pipeline으로 구성해 학습·평가하고, 평가지표(정확도·정밀도·재현율·
#           F1·ROC-AUC)를 출력한 뒤 joblib으로 모델을 저장합니다.
# 작성일: 2026-07-21
# 프로그램명: End2End 데이터 분석 프로젝트 - ML Pipeline 모듈
# 변경사항 내역:
# - 2026-07-21, 최초 작성
# - 2026-07-21, 날짜 피처(_hour/_dayofweek/_day/_month) 생성 코드가 for 루프
#               밖에 있어 여러 datetime 컬럼 중 마지막 컬럼만 처리되고 나머지는
#               원본 datetime 타입으로 남아 OneHotEncoder로 잘못 넘어가던 버그
#               수정 (들여쓰기를 루프 안으로 이동)
#====================================================================

import logging
import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

logger = logging.getLogger(__name__)


def _build_target_and_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    타깃과 입력 피처를 구성합니다.

    우선순위:
    1. payment_type
    2. is_high_fare
    3. target
    4. fare_amount 중앙값 기준 이진 타깃 생성
    """
    target_candidates = ["payment_type", "is_high_fare", "target"]

    working_df = df.copy()

    target_col = None
    for col in target_candidates:
        if col in working_df.columns:
            target_col = col
            break

    # 타깃 컬럼이 없으면 fare_amount 중앙값을 기준으로 생성
    if target_col is None:
        if "fare_amount" not in working_df.columns:
            raise ValueError(
                "타깃 컬럼이 없고 fare_amount도 없어 타깃을 만들 수 없습니다."
            )

        median_fare = working_df["fare_amount"].median()
        working_df["target"] = (
            working_df["fare_amount"] >= median_fare
        ).astype(int)
        target_col = "target"

    # 타깃 결측치 제거
    working_df = working_df.dropna(subset=[target_col]).copy()

    y = working_df[target_col]
    X = working_df.drop(columns=[target_col])

    # 결제 방식 또는 요금 등 결과를 예측할 때 데이터 누수를 일으킬 수 있는 컬럼 제거
    leakage_cols = [
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
        "congestion_surcharge",
        "Airport_fee",
        "cbd_congestion_fee",
    ]
    X = X.drop(columns=leakage_cols, errors="ignore")

    # 날짜·시간 컬럼에서 분석에 의미 있는 시간 파생 변수 생성
    datetime_cols = [
        col for col in X.columns
        if pd.api.types.is_datetime64_any_dtype(X[col])
    ]

    # [버그 수정] 아래 피처 생성 코드가 원래 for 루프 밖(들여쓰기 오류)에 있어서
    # datetime_cols가 여러 개일 때 마지막 컬럼만 처리되고, 나머지는 원본
    # datetime 타입 그대로 남아 있었다. 이 상태로 _build_pipeline()에 넘어가면
    # datetime이 "숫자가 아니다"라는 이유로 범주형(OneHotEncoder) 대상으로
    # 잘못 분류되어, 값마다(수백만 개) 별도 카테고리를 만들려 시도해 메모리
    # 폭발이나 극심한 학습 지연으로 이어질 수 있었다. 각 반복마다 4개 피처
    # 생성 + 원본 컬럼 삭제까지 전부 루프 "안"에서 끝나도록 들여쓰기를 수정했다.
    for col in datetime_cols:
        dt_col = pd.to_datetime(X[col], errors="coerce")

        # 큰 UNIX timestamp 대신 시간 특징을 생성
        X[f"{col}_hour"] = dt_col.dt.hour
        X[f"{col}_dayofweek"] = dt_col.dt.dayofweek
        X[f"{col}_day"] = dt_col.dt.day
        X[f"{col}_month"] = dt_col.dt.month

        # 원본 날짜 컬럼은 제거
        X = X.drop(columns=[col])

    # 무한대 값을 결측치로 변경
    X = X.replace([np.inf, -np.inf], np.nan)

    # 숫자형 극단값을 0.1%~99.9% 범위로 제한
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns

    for col in numeric_cols:
        lower = X[col].quantile(0.001)
        upper = X[col].quantile(0.999)

        if pd.notna(lower) and pd.notna(upper):
            X[col] = X[col].clip(lower=lower, upper=upper)

    return X, y, target_col


def _build_pipeline(X: pd.DataFrame) -> Pipeline:
    """숫자형·범주형 전처리와 SGD 분류기를 포함한 Pipeline 생성"""

    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = [
        col for col in X.columns
        if col not in numeric_features
    ]

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]
        )

        transformers.append(
            ("num", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        transformers.append(
            ("cat", categorical_pipeline, categorical_features)
        )

    if not transformers:
        raise ValueError("학습에 사용할 피처가 없습니다.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def train_evaluate_and_save_model(
    df: pd.DataFrame,
    model_path: str,
) -> Dict[str, float]:
    """
    Pipeline 기반으로 전처리·학습·평가를 수행하고 모델을 저장합니다.
    """

    X, y, target_col = _build_target_and_features(df)

    if y.nunique() < 2:
        raise ValueError(
            "타깃 클래스가 1개뿐이라 분류 모델 학습이 불가능합니다."
        )

    # 데이터가 너무 적으면 분할 자체가 불가능하므로 안내 후 전체 데이터 사용
    train_on_full = len(X) < 10 or y.value_counts().min() < 2

    if train_on_full:
        logger.warning(
            "데이터가 매우 적어 학습/평가를 동일 데이터에서 수행합니다. "
            "이 평가지표는 실제 성능으로 해석할 수 없습니다."
        )
        X_train, X_test, y_train, y_test = X, X, y, y

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

    pipeline = _build_pipeline(X)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # 클래스가 2개면 binary, 3개 이상이면 weighted 평균 사용
    is_binary = y.nunique() == 2
    avg_type = "binary" if is_binary else "weighted"

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            average=avg_type,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            average=avg_type,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            average=avg_type,
            zero_division=0,
        ),
    }

    # ROC-AUC: 이진/다중 분류 모두 계산 시도
    try:
        if is_binary:
            metrics["roc_auc"] = roc_auc_score(
                y_test,
                y_proba[:, 1],
            )
        else:
            classes = pipeline.named_steps["model"].classes_

            metrics["roc_auc"] = roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr",
                average="weighted",
                labels=classes,
            )
    except ValueError as exc:
        logger.warning(f"ROC-AUC 계산 생략: {exc}")

    # models/ 폴더 생성
    model_dir = os.path.dirname(model_path)

    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    # 전처리 + 모델 전체 저장
    joblib.dump(pipeline, model_path)

    logger.info("=" * 60)
    logger.info("[10] ML Pipeline 학습 및 평가")
    logger.info("=" * 60)
    logger.info(f"▶ 타깃 컬럼: {target_col}")
    logger.info(f"▶ 클래스 수: {y.nunique()}개")
    logger.info(f"▶ accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"▶ precision: {metrics['precision']:.4f}")
    logger.info(f"▶ recall:    {metrics['recall']:.4f}")
    logger.info(f"▶ f1:        {metrics['f1']:.4f}")

    if "roc_auc" in metrics:
        logger.info(f"▶ roc_auc:   {metrics['roc_auc']:.4f}")

    logger.info(f"▶ 모델 저장 완료: {model_path}")

    return metrics