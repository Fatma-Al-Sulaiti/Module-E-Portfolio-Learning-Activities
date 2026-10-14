# Machine Learning (ML_PCOM7E) – e-Portfolio Repository

Code and written documents for each e-portfolio activity. Every activity has a **Jupyter notebook** (executed, outputs saved) and a separate **Word document** that explains the aim, method, results and reflection.

| Folder | Activity | Code | Written document |
|---|---|---|---|
| `unit-02-eda/` | Unit 2 – Exploratory Data Analysis | `unit02_exploratory_data_analysis.ipynb` | `Unit02_EDA_written_report.docx` |
| `unit-03-correlation-regression/` | Unit 3 – Covariance & Pearson correlation | `covariance_pearson_correlation.ipynb` | `Covariance_Pearson_correlation_written_report.docx` |
| | Unit 3 – Simple linear regression | `linear_regression.ipynb` | `Linear_regression_written_report.docx` |
| | Unit 3 – Multiple linear regression | `multiple_linear_regression.ipynb` | `Multiple_linear_regression_written_report.docx` |
| | Unit 3 – Polynomial regression | `polynomial_regression.ipynb` | `Polynomial_regression_written_report.docx` |
| `unit-05-clustering-jaccard/` | Unit 5 – Jaccard coefficient | `jaccard_coefficient.ipynb` | `Jaccard_coefficient_written_report.docx` |
| `projects/airbnb-nyc/` | Group 5 Airbnb NYC project (Appendix A code) | `airbnb_price_prediction_segmentation.py` | see the group report |
| `eportfolio/` | Final e-portfolio (Parts A and B) | – | `ML_PCOM7E_ePortfolio_Parts_A_B.docx` |

## Running
```bash
pip install -r requirements.txt
jupyter notebook          # open any notebook and run all cells
```
The notebooks use datasets bundled with scikit-learn or simulated data, so they run offline in Jupyter or Google Colab. Variables at the top of each notebook (noise, sample size, feature lists) can be changed to observe their effect.

The Airbnb script needs `AB_NYC_2019.csv` (Dgomonov, 2019, Kaggle) in the same folder as the script (it is not committed). It takes about 5–8 minutes and writes four figures to `figures/`; `sample_run_output.txt` is the output from a full run.
