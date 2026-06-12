"""CRISP-DM analysis for the Kaggle 50 Startups dataset."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "50_Startups.csv"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

NUMERIC_FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
CATEGORICAL_FEATURES = ["State"]
TARGET = "Profit"


def adjusted_r2(r2: float, sample_count: int, feature_count: int) -> float:
    denominator = sample_count - feature_count - 1
    if denominator <= 0:
        return float("nan")
    return 1 - (1 - r2) * (sample_count - 1) / denominator


def save_eda(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include=np.number)
    correlation = numeric.corr()
    correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv", encoding="utf-8-sig")

    quality = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isna().sum(),
            "unique_count": df.nunique(),
        }
    )
    quality.loc["__dataset__", "dtype"] = "summary"
    quality.loc["__dataset__", "missing_count"] = int(df.duplicated().sum())
    quality.loc["__dataset__", "unique_count"] = len(df)
    quality.to_csv(OUTPUT_DIR / "data_quality.csv", encoding="utf-8-sig")

    df.describe(include="all").transpose().to_csv(
        OUTPUT_DIR / "descriptive_statistics.csv", encoding="utf-8-sig"
    )
    df.groupby("State")[TARGET].agg(["count", "mean", "median", "std"]).to_csv(
        OUTPUT_DIR / "state_profit_summary.csv", encoding="utf-8-sig"
    )

    plt.figure(figsize=(7, 5))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, feature in zip(axes, NUMERIC_FEATURES):
        sns.regplot(data=df, x=feature, y=TARGET, ax=ax, scatter_kws={"s": 28})
        ax.set_title(f"{feature} vs Profit")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "numeric_features_vs_profit.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="State", y=TARGET)
    plt.title("Profit Distribution by State")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "state_vs_profit.png", dpi=180)
    plt.close()


def encode_train_test(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    train_states = encoder.fit_transform(x_train[CATEGORICAL_FEATURES])
    test_states = encoder.transform(x_test[CATEGORICAL_FEATURES])
    state_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()

    train_encoded = pd.concat(
        [
            x_train[NUMERIC_FEATURES].reset_index(drop=True),
            pd.DataFrame(train_states, columns=state_names),
        ],
        axis=1,
    )
    test_encoded = pd.concat(
        [
            x_test[NUMERIC_FEATURES].reset_index(drop=True),
            pd.DataFrame(test_states, columns=state_names),
        ],
        axis=1,
    )
    return train_encoded, test_encoded


def correlation_selection(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[list[str], dict]:
    correlations = x_train.apply(lambda col: col.corr(y_train)).sort_values(
        key=lambda values: values.abs(), ascending=False
    )
    selected = correlations[correlations.abs() >= 0.5].index.tolist()
    if not selected:
        selected = [correlations.abs().idxmax()]
    return selected, {"correlations": correlations.to_dict(), "threshold": 0.5}


def kbest_selection(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[list[str], dict]:
    selector = SelectKBest(score_func=f_regression, k=min(3, x_train.shape[1]))
    selector.fit(x_train, y_train)
    selected = x_train.columns[selector.get_support()].tolist()
    details = {
        name: {"f_score": float(score), "p_value": float(p_value)}
        for name, score, p_value in zip(x_train.columns, selector.scores_, selector.pvalues_)
    }
    return selected, details


def backward_selection(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[list[str], dict]:
    selected = x_train.columns.tolist()
    removal_history: list[dict] = []
    while selected:
        model = sm.OLS(y_train, sm.add_constant(x_train[selected], has_constant="add")).fit()
        feature_pvalues = model.pvalues.drop("const")
        worst_feature = feature_pvalues.idxmax()
        worst_pvalue = float(feature_pvalues.max())
        if worst_pvalue <= 0.05:
            break
        removal_history.append({"feature": worst_feature, "p_value": worst_pvalue})
        selected.remove(worst_feature)

    final_model = sm.OLS(
        y_train, sm.add_constant(x_train[selected], has_constant="add")
    ).fit()
    return selected, {
        "threshold": 0.05,
        "removal_history": removal_history,
        "final_p_values": final_model.pvalues.to_dict(),
    }


def rfe_selection(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[list[str], dict]:
    selector = RFE(
        estimator=LinearRegression(), n_features_to_select=min(3, x_train.shape[1])
    )
    selector.fit(x_train, y_train)
    selected = x_train.columns[selector.support_].tolist()
    return selected, dict(zip(x_train.columns, selector.ranking_.astype(int).tolist()))


def lasso_selection(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[list[str], dict]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x_train)
    selector = LassoCV(cv=CV_FOLDS, random_state=RANDOM_STATE, max_iter=100_000)
    selector.fit(scaled, y_train)
    coefficients = pd.Series(selector.coef_, index=x_train.columns)
    selected = coefficients[coefficients.abs() > 1e-8].index.tolist()
    if not selected:
        selected = [coefficients.abs().idxmax()]
    return selected, {
        "alpha": float(selector.alpha_),
        "standardized_coefficients": coefficients.to_dict(),
    }


def evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    selectors = [
        ("Model_1", "Correlation-based Feature Selection", correlation_selection),
        ("Model_2", "SelectKBest / f_regression", kbest_selection),
        ("Model_3", "Backward Elimination", backward_selection),
        ("Model_4", "Recursive Feature Elimination", rfe_selection),
        ("Model_5", "Lasso-based Feature Selection", lasso_selection),
    ]
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    results: list[dict] = []
    coefficient_rows: list[dict] = []
    prediction_frame = pd.DataFrame(
        {"row_index": y_test.index, "actual_profit": y_test.to_numpy()}
    )
    selection_details: dict = {}

    for model_id, method, selector_fn in selectors:
        selected, details = selector_fn(x_train, y_train.reset_index(drop=True))
        selection_details[model_id] = {
            "method": method,
            "selected_features": selected,
            "details": details,
        }
        model = LinearRegression()
        model.fit(x_train[selected], y_train)
        predictions = model.predict(x_test[selected])
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        cv_scores = cross_validate(model, x_train[selected], y_train, cv=cv, scoring=scoring)
        results.append(
            {
                "model_id": model_id,
                "feature_selection_method": method,
                "selected_features": ", ".join(selected),
                "feature_count": len(selected),
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "Adjusted_R2": adjusted_r2(r2, len(y_test), len(selected)),
                "CV_MAE_mean": -cv_scores["test_mae"].mean(),
                "CV_RMSE_mean": -cv_scores["test_rmse"].mean(),
                "CV_R2_mean": cv_scores["test_r2"].mean(),
                "interpretation": "Lower test RMSE/MAE is better; use CV as a stability check.",
            }
        )
        coefficient_rows.append(
            {
                "model_id": model_id,
                "feature": "Intercept",
                "coefficient": model.intercept_,
            }
        )
        coefficient_rows.extend(
            {
                "model_id": model_id,
                "feature": feature,
                "coefficient": coefficient,
            }
            for feature, coefficient in zip(selected, model.coef_)
        )
        prediction_frame[model_id] = predictions

    result_frame = pd.DataFrame(results).sort_values(
        ["RMSE", "MAE", "Adjusted_R2"], ascending=[True, True, False]
    )
    return result_frame, pd.DataFrame(coefficient_rows), prediction_frame, selection_details


def save_model_figures(results: pd.DataFrame, predictions: pd.DataFrame) -> None:
    best_id = results.iloc[0]["model_id"]
    actual = predictions["actual_profit"]
    predicted = predictions[best_id]

    plt.figure(figsize=(7, 5))
    sns.barplot(data=results, x="model_id", y="RMSE", hue="model_id", legend=False)
    plt.title("Test RMSE by Model")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_rmse_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, alpha=0.8)
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    plt.plot([low, high], [low, high], linestyle="--", color="black")
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    plt.title(f"Actual vs Predicted Profit ({best_id})")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "best_model_actual_vs_predicted.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    residuals = actual - predicted
    plt.scatter(predicted, residuals, alpha=0.8)
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Predicted Profit")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot ({best_id})")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "best_model_residuals.png", dpi=180)
    plt.close()


def write_report(df: pd.DataFrame, results: pd.DataFrame, details: dict) -> None:
    best = results.iloc[0]
    corr = df.select_dtypes(include=np.number).corr()[TARGET].drop(TARGET)
    state_means = df.groupby("State")[TARGET].mean().sort_values(ascending=False)
    model_lines = "\n".join(
        f"| {row.model_id} | {row.feature_selection_method} | {row.selected_features} | "
        f"{row.MAE:,.2f} | {row.RMSE:,.2f} | {row.R2:.4f} | {row.Adjusted_R2:.4f} | "
        f"{row.CV_RMSE_mean:,.2f} |"
        for row in results.itertuples()
    )
    report = f"""# 50 Startups：CRISP-DM 線性迴歸分析報告

## 1. Business Understanding

本分析以 `Profit` 為預測目標，只比較 Linear Regression 搭配不同特徵選擇方法。
核心問題是找出哪些投入最能解釋公司利潤，並建立可解釋、可重現的預測模型。

## 2. Data Understanding

- 資料筆數：{len(df)}
- 原始特徵數：4（3 個數值特徵、1 個類別特徵）
- 缺失值：{int(df.isna().sum().sum())}
- 重複列：{int(df.duplicated().sum())}
- State 分布：{", ".join(f"{k}={v}" for k, v in df["State"].value_counts().items())}

與 Profit 的 Pearson 相關係數：

- R&D Spend：{corr["R&D Spend"]:.4f}
- Marketing Spend：{corr["Marketing Spend"]:.4f}
- Administration：{corr["Administration"]:.4f}

各州平均 Profit：{", ".join(f"{k}={v:,.2f}" for k, v in state_means.items())}。
州別平均值有差異，但這不代表因果效果，仍需由模型與更多資料驗證。

## 3. Data Preparation

- `State` 使用 One-Hot Encoding，並設定 `drop="first"` 避免 dummy variable trap。
- 先以 `random_state=42` 將資料切成 80% 訓練集與 20% 測試集。
- 編碼器與所有特徵選擇器只在訓練集上 fitting，以控制資料洩漏。
- Lasso 特徵選擇前使用 StandardScaler；最終預測模型仍為 Linear Regression。
- 使用 shuffled 5-fold K-Fold Cross Validation 檢查訓練集上的模型穩定度。

## 4. Modeling 與 Evaluation

| Model | 特徵選擇方法 | 選中特徵 | Test MAE | Test RMSE | Test R2 | Adjusted R2 | 5-fold CV RMSE |
|---|---|---|---:|---:|---:|---:|---:|
{model_lines}

依設計規則，以測試集 RMSE、MAE 為主要依據，Adjusted R2 為次要依據。
目前最佳模型是 **{best["model_id"]}（{best["feature_selection_method"]}）**，
選中特徵為 **{best["selected_features"]}**，Test RMSE 為 **{best["RMSE"]:,.2f}**，
Test MAE 為 **{best["MAE"]:,.2f}**，Test R2 為 **{best["R2"]:.4f}**。

## 5. Business Interpretation

- R&D Spend 與 Profit 呈高度正相關，且被多數方法選中，是最穩定的重要預測因子。
- Marketing Spend 與 Profit 有中高度相關，但它也與 R&D Spend 高度相關，因此納入多變量模型後的額外貢獻可能下降。
- Administration 與 Profit 的單變量相關性低，對預測的幫助有限。
- State 是否入選不能被解讀為地點造成利潤差異；樣本僅 50 筆，且未控制產業、公司規模等混淆因素。

## 6. Deployment 與限制

此結果適合作為教學、探索性分析與初步預算討論依據，不宜直接作為高風險投資決策。
資料只有 50 筆，單次測試集只有 {len(df) * TEST_SIZE:.0f} 筆，測試指標容易受切分影響；
實際部署前應增加樣本、監控輸入範圍、定期重新訓練，並保留人工審查。

## 輸出檔案

- `outputs/model_comparison.csv`：五個模型的測試與交叉驗證結果
- `outputs/model_coefficients.csv`：各模型係數
- `outputs/test_predictions.csv`：測試集實際值與預測值
- `outputs/feature_selection_details.json`：特徵選擇細節
- `outputs/figures/`：EDA 與模型評估圖表
"""
    (ROOT / "50_startups_analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    if set(df.columns) != expected:
        raise ValueError(f"Unexpected columns: {df.columns.tolist()}")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values; define an imputation strategy first.")

    save_eda(df)
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    x_train, x_test = encode_train_test(x_train_raw, x_test_raw)
    y_train = y_train.reset_index(drop=True)

    results, coefficients, predictions, details = evaluate_models(
        x_train, x_test, y_train, y_test
    )
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(
        OUTPUT_DIR / "model_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    predictions.to_csv(
        OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "feature_selection_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_model_figures(results, predictions)
    write_report(df, results, details)

    print(results.to_string(index=False))
    print(f"\nBest model: {results.iloc[0]['model_id']}")
    print(f"Report: {ROOT / '50_startups_analysis_report.md'}")


if __name__ == "__main__":
    main()
