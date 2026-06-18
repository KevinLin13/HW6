"""Export the executed learning notebook as Markdown, HTML, and named figures."""

from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "50_startups_crispdm_learning.ipynb"
REPORT_DIR = ROOT / "reports"
REPORT_NAME = "50_startups_complete_report"
ASSET_DIR = REPORT_DIR / f"{REPORT_NAME}_files"

FIGURE_NAMES = {
    f"{REPORT_NAME}_12_1.png": "01_correlation_and_rd_profit.png",
    f"{REPORT_NAME}_44_0.png": "02_lasso_cv_alpha.png",
    f"{REPORT_NAME}_63_0.png": "03_candidate_cv_rmse_distribution.png",
    f"{REPORT_NAME}_68_0.png": "04_candidate_model_evidence.png",
    f"{REPORT_NAME}_79_0.png": "05_pipeline_cv_feature_count_comparison.png",
    f"{REPORT_NAME}_90_0.png": "06_final_model_diagnostics.png",
}


def export(format_name):
    subprocess.run(
        [
            "python",
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            format_name,
            "--no-input",
            "--output",
            REPORT_NAME,
            "--output-dir",
            str(REPORT_DIR),
            str(NOTEBOOK),
        ],
        cwd=ROOT,
        check=True,
    )


REPORT_DIR.mkdir(exist_ok=True)
if ASSET_DIR.exists():
    shutil.rmtree(ASSET_DIR)

export("markdown")
export("html")

markdown_path = REPORT_DIR / f"{REPORT_NAME}.md"
markdown = markdown_path.read_text(encoding="utf-8")
for generated_name, descriptive_name in FIGURE_NAMES.items():
    source = ASSET_DIR / generated_name
    destination = ASSET_DIR / descriptive_name
    source.rename(destination)
    markdown = markdown.replace(generated_name, descriptive_name)

markdown_path.write_text(markdown, encoding="utf-8")
print(f"Exported report: {markdown_path}")
print(f"Exported report: {REPORT_DIR / f'{REPORT_NAME}.html'}")

