# 50 Startups：CRISP-DM 與 Linear Regression 學習專案

本專案使用 Kaggle `50 Startups` 資料集，透過可完整執行的
[`50_startups_crispdm_learning.ipynb`](50_startups_crispdm_learning.ipynb)，
依循 CRISP-DM 流程學習資料理解、資料準備、特徵篩選、Linear Regression、
Cross-Validation、模型評估與簡易部署。

Notebook 已保存所有表格與圖形輸出，可直接開啟閱讀。

## 完整圖文報告

- [報告索引與圖表清單](reports/README.md)
- [完整 Markdown 圖文報告](reports/50_startups_complete_report.md)
- [完整 HTML 圖文報告](reports/50_startups_complete_report.html)

報告由已執行 Notebook 匯出，隱藏程式碼並保留所有教學文字、結果表格與圖形。
六張圖表亦已獨立儲存於 `reports/50_startups_complete_report_files/`。

## 專案檔案

| 檔案 | 用途 |
|---|---|
| `50_startups_crispdm_learning.ipynb` | 主要學習 Notebook，包含 25 個循序學習問題與執行結果 |
| `50_Startups.csv` | Notebook 使用的原始資料集 |
| `generate_learning_notebook.py` | 重新產生學習 Notebook 的來源程式 |
| `export_learning_report.py` | 將已執行 Notebook 匯出為完整圖文報告 |
| `requirements.txt` | 執行 Notebook 所需套件 |
| `learning.txt` | 原始學習需求 |
| `Prompt.txt` | 專案提示內容 |
| `LOG_PROMPT.txt` | 提示紀錄 |
| `reports/` | 完整 Markdown／HTML 圖文報告與獨立圖表 |

## 執行方式

```powershell
python -m pip install -r requirements.txt
python generate_learning_notebook.py
```

產生 Notebook 後，可使用 Jupyter 或支援 Notebook 的 IDE 開啟：

```powershell
jupyter lab 50_startups_crispdm_learning.ipynb
```

## 資料集與預測目標

資料集共有 50 筆新創公司資料：

| 欄位 | 類型 | 說明 |
|---|---|---|
| `R&D Spend` | 數值 | 研發支出 |
| `Administration` | 數值 | 行政支出 |
| `Marketing Spend` | 數值 | 行銷支出 |
| `State` | 類別 | 公司所在州別 |
| `Profit` | 數值 | 預測目標 |

資料沒有缺失值與重複資料。初步探索顯示：

- `R&D Spend` 與 `Profit` 的線性關係最強。
- `Marketing Spend` 與 `Profit` 有中度至高度正相關。
- `Administration` 的單變量線性關係較弱。
- 州別平均 Profit 有差異，但不能解讀為州別造成 Profit 差異。

## CRISP-DM 流程

1. **Business Understanding：** 根據支出與州別預測 Profit。
2. **Data Understanding：** 檢查資料品質、分布、相關性與可能限制。
3. **Data Preparation：** 先切分 Train/Test，再以 Training Set 擬合編碼器與篩選器。
4. **Modeling：** 使用五種特徵篩選方法搭配 Linear Regression。
5. **Evaluation：** 使用 Pipeline CV、Repeated CV 與最終外部 Test Set。
6. **Deployment：** 提供接收原始欄位並預測 Profit 的函式。

## 25 個學習問題

Notebook 將內容整理為以下主題：

1. 商業目標與預測問題
2. 特徵 `X` 與目標 `y`
3. 數值型與類別型特徵
4. 建模前的資料觀察
5. 初步重要特徵
6. Training/Test Split
7. Data Leakage
8. 完整資料準備流程
9. 特徵篩選概念
10. Correlation-based Selection
11. SelectKBest 與 `f_regression`
12. Backward Elimination
13. Recursive Feature Elimination
14. Lasso-based Selection
15. 五種方法選取結果整理
16. Linear Regression 與係數解讀
17. MAE、MSE、RMSE、R² 與 Adjusted R²
18. 本專案的模型選擇流程
19. Cross-Validation 與小樣本解讀
20. 固定特徵 CV 與 Pipeline CV 的差異
21. Pipeline + CV 公平比較
22. 單次 CV 與 Repeated CV 的敏感度
23. 外部 Test Set 最終評估
24. 簡易部署
25. 專案總結與限制

## 五種特徵篩選方法

| 方法 | 核心概念 | 主要限制 |
|---|---|---|
| Correlation | 依特徵與 Profit 的單變量相關性篩選 | 忽略多變量條件效果與交互作用 |
| SelectKBest | 依 `f_regression` 分數選前 `k` 個特徵 | 屬於單變量評分，結果受 `k` 影響 |
| Backward Elimination | 依多變量 OLS p-value 逐步移除特徵 | 小樣本下 p-value 與選取結果可能不穩定 |
| RFE | 反覆訓練模型並移除較弱特徵 | 結果依賴模型與指定特徵數 |
| Lasso | 使用 L1 懲罰壓縮係數 | 需標準化；相關特徵可能互相競爭 |

不同方法可能選到相同特徵。若後續使用同一個 Linear Regression，相同特徵集合
會產生相同預測與評估結果，這不是程式錯誤。

## 正確的模型比較流程

本專案不使用外部 Test Set 反覆挑選模型，而採用以下流程：

1. 只用 Training Set 進行合理的特徵探索。
2. 比較少量且有理論依據的候選組合。
3. 將特徵篩選器與 Linear Regression 放入同一個 Pipeline。
4. 使用 5-Fold CV 比較平均表現與切分波動。
5. 使用 Repeated 5-Fold CV 檢查結果是否依賴單一切分。
6. 綜合效能、穩定性、簡潔性與可解釋性選定模型。
7. 最後才使用外部 Test Set 評估一次。

### 為什麼需要 Pipeline + CV？

若先使用完整 Training Set 選特徵，再進行 CV，Validation Fold 已間接參與特徵
選擇，分數可能稍微樂觀。Pipeline 可確保每一折只使用 Fold Training Data
擬合特徵篩選器與模型，再以 Validation Fold 評分。

### 為什麼還需要 Repeated CV？

原始資料只有 50 筆；保留 10 筆外部 Test Set 後，5-Fold CV 的每折 Validation
Data 只有約 8 筆。少數觀測值就可能改變 RMSE 與方法排名，因此需用不同切分
檢查結果敏感度。

## 模型比較結果

### 單次 Pipeline + 5-Fold CV

單次切分中，`RFE, k=3` 與 `Backward Elimination, k=3` 的平均 Validation RMSE
最低。兩者在五折中選到相同三個數值特徵，因此建立相同模型並得到相同分數。

### Repeated 5-Fold CV

重複不同切分後，每種方法的最佳設定如下：

| 方法 | 最佳特徵數 | Repeated CV RMSE Mean | Repeated CV RMSE Std |
|---|---:|---:|---:|
| Correlation | 2 | 9,644.40 | 3,170.90 |
| SelectKBest | 2 | 9,644.40 | 3,170.90 |
| Lasso | 2 | 9,652.86 | 3,185.93 |
| RFE | 2 | 9,740.42 | 3,250.35 |
| Backward Elimination | 1 | 9,862.04 | 3,446.15 |

Correlation 與 SelectKBest 表現相同，是因為兩者都選到
`R&D Spend + Marketing Spend`。這不表示 Correlation 是普遍最佳方法，而是此
資料主要由少數強線性特徵驅動。

## 最終模型

單次 CV 與 Repeated CV 的方法排名並不一致，因此沒有足夠證據宣告某一種特徵
篩選方法穩定勝出。Notebook 最終採用多種方法共同支持的：

### Consensus Two-Feature Model

```text
特徵：R&D Spend、Marketing Spend
```

選擇理由：

- 兩個特徵在多種方法與不同切分中反覆被選取。
- Repeated CV 表現合理。
- 模型簡潔且容易解釋。
- 不將結果錯誤歸因為某一個篩選方法獲勝。

最終評估結果：

| 指標 | 結果 |
|---|---:|
| 5-Fold CV MAE Mean | 約 7,058 |
| 5-Fold CV RMSE Mean | 約 9,187 |
| 5-Fold CV RMSE Std | 約 3,059 |
| 外部 Test RMSE | 約 8,206 |

外部 Test Set 僅在完整流程選定後使用一次，不參與模型或特徵數選擇。

## 如何解讀最終模型？

- `R&D Spend` 與 Profit 有最強的正向線性預測關係。
- `Marketing Spend` 提供額外的預測資訊，但係數不代表因果效果。
- 被排除的特徵不代表沒有商業價值，只表示在目前資料與線性模型中，額外預測
  貢獻有限或不穩定。
- 模型結果不能直接轉換成預算配置建議。

## 專案限制

- 資料只有 50 筆，CV 與 Test 分數都有高度抽樣不確定性。
- 不同 CV 切分可能改變方法排名。
- 線性模型無法自動捕捉複雜非線性與交互作用。
- 特徵選擇與模型係數呈現關聯，而非因果效果。
- 真實部署前應蒐集更多產業、公司年齡、員工數與時間資料。

## 最重要的結論

本專案沒有證明某一種特徵篩選方法永遠最佳。較準確的結論是：

> 在目前 50 筆資料、Linear Regression 與小樣本驗證限制下，多種方法共同支持
> `R&D Spend + Marketing Spend` 作為合理、簡潔且可解釋的最終特徵組合。
