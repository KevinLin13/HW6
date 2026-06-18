# 50 Startups 完整圖文報告

本目錄彙整 `50_startups_crispdm_learning.ipynb` 的教學文字、分析結果、表格與圖形。
報告隱藏程式碼輸入，適合直接閱讀或提交。

## 報告格式

- [完整 Markdown 圖文報告](50_startups_complete_report.md)
- [完整 HTML 圖文報告](50_startups_complete_report.html)

HTML 版適合直接使用瀏覽器閱讀；Markdown 版適合 GitHub、IDE 與版本控制。

## 圖表

| 圖表 | 說明 |
|---|---|
| [01_correlation_and_rd_profit.png](50_startups_complete_report_files/01_correlation_and_rd_profit.png) | 數值欄位相關係數與 R&D Spend 對 Profit 關係 |
| [02_lasso_cv_alpha.png](50_startups_complete_report_files/02_lasso_cv_alpha.png) | LassoCV alpha 選擇 |
| [03_candidate_cv_rmse_distribution.png](50_startups_complete_report_files/03_candidate_cv_rmse_distribution.png) | 候選模型逐折 CV RMSE 分布 |
| [04_candidate_model_evidence.png](50_startups_complete_report_files/04_candidate_model_evidence.png) | 候選模型表現、波動、簡潔性與排名證據 |
| [05_pipeline_cv_feature_count_comparison.png](50_startups_complete_report_files/05_pipeline_cv_feature_count_comparison.png) | Pipeline CV 下方法與特徵數比較 |
| [06_final_model_diagnostics.png](50_startups_complete_report_files/06_final_model_diagnostics.png) | 最終模型 Actual vs Predicted 與 Residual Plot |

## 重新匯出

```powershell
python export_learning_report.py
```
