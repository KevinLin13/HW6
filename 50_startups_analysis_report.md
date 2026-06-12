# 50 Startups：CRISP-DM 線性迴歸分析報告

## 1. Business Understanding

本分析以 `Profit` 為預測目標，只比較 Linear Regression 搭配不同特徵選擇方法。
核心問題是找出哪些投入最能解釋公司利潤，並建立可解釋、可重現的預測模型。

## 2. Data Understanding

- 資料筆數：50
- 原始特徵數：4（3 個數值特徵、1 個類別特徵）
- 缺失值：0
- 重複列：0
- State 分布：New York=17, California=17, Florida=16

與 Profit 的 Pearson 相關係數：

- R&D Spend：0.9729
- Marketing Spend：0.7478
- Administration：0.2007

各州平均 Profit：Florida=118,774.02, New York=113,756.45, California=103,905.18。
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
| Model_3 | Backward Elimination | R&D Spend | 6,077.36 | 7,714.33 | 0.9265 | 0.9173 | 9,406.58 |
| Model_4 | Recursive Feature Elimination | R&D Spend, State_Florida, State_New York | 6,296.78 | 7,946.37 | 0.9220 | 0.8830 | 10,007.91 |
| Model_1 | Correlation-based Feature Selection | R&D Spend, Marketing Spend | 6,469.18 | 8,206.33 | 0.9168 | 0.8931 | 9,186.68 |
| Model_2 | SelectKBest / f_regression | R&D Spend, Marketing Spend, State_New York | 6,430.58 | 8,242.78 | 0.9161 | 0.8741 | 9,460.92 |
| Model_5 | Lasso-based Feature Selection | R&D Spend, Administration, Marketing Spend | 6,979.15 | 8,995.91 | 0.9001 | 0.8501 | 9,104.68 |

依設計規則，以測試集 RMSE、MAE 為主要依據，Adjusted R2 為次要依據。
目前最佳模型是 **Model_3（Backward Elimination）**，
選中特徵為 **R&D Spend**，Test RMSE 為 **7,714.33**，
Test MAE 為 **6,077.36**，Test R2 為 **0.9265**。

## 5. Business Interpretation

- R&D Spend 與 Profit 呈高度正相關，且被多數方法選中，是最穩定的重要預測因子。
- Marketing Spend 與 Profit 有中高度相關，但它也與 R&D Spend 高度相關，因此納入多變量模型後的額外貢獻可能下降。
- Administration 與 Profit 的單變量相關性低，對預測的幫助有限。
- State 是否入選不能被解讀為地點造成利潤差異；樣本僅 50 筆，且未控制產業、公司規模等混淆因素。

## 6. Deployment 與限制

此結果適合作為教學、探索性分析與初步預算討論依據，不宜直接作為高風險投資決策。
資料只有 50 筆，單次測試集只有 10 筆，測試指標容易受切分影響；
實際部署前應增加樣本、監控輸入範圍、定期重新訓練，並保留人工審查。

## 輸出檔案

- `outputs/model_comparison.csv`：五個模型的測試與交叉驗證結果
- `outputs/model_coefficients.csv`：各模型係數
- `outputs/test_predictions.csv`：測試集實際值與預測值
- `outputs/feature_selection_details.json`：特徵選擇細節
- `outputs/figures/`：EDA 與模型評估圖表
