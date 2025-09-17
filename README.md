# Retail Sales Forecast — Linear Regression (scikit-learn)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/) 
[![pandas](https://img.shields.io/badge/pandas-DataFrame-green?logo=pandas&logoColor=white)](https://pandas.pydata.org/) 
[![NumPy](https://img.shields.io/badge/NumPy-Numerical-orange?logo=numpy&logoColor=white)](https://numpy.org/) 
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) 
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red?logo=plotly&logoColor=white)](https://matplotlib.org/)

Minimal project for monthly sales forecasting using **Linear Regression** with scikit-learn. Includes basic data handling, December forecast, and exploratory visualizations.

## 📂 Project Structure

```
.
├── notebooks/                                  # optional Jupyter exploration
│   └── 01\_exploration.ipynb
└── reports/                                    # outputs generated automatically
    └── figures/
        ├── histogram\_sales.png
        └── scatter\_month\_vs\_sales.png
├── .gitignore 
├── main.py                                     # main script (training, forecast, plots)
├── README.md
└── requirements.txt                            # dependencies
```

> **Note:** the folders `reports/` and `reports/figures/` are created automatically by the code if they don’t exist.  

## ⚙️ Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
## 🚀 Usage

**Default forecast (OLS - linear regression):**

```bash
python main.py
```

**Naive forecast (constant step):**

```bash
python main.py --method naive
```

**Custom output directory:**

```bash
python main.py --output-dir out/figs
```

Example output:

```
December forecast (OLS (linear regression)): 3356.36
Figures saved to:
 - reports/figures/histogram_sales.png
 - reports/figures/scatter_month_vs_sales.png
```

## 📊 Outputs

* `histogram_sales.png` → Sales histogram
* `scatter_month_vs_sales.png` → Month vs Sales scatter plot with regression line

## 🧭 Technical Decisions

* **Reproducibility:** output directories are created automatically.
* **Simplicity:** single script covers the full flow (data → training → forecast → plots).
* **Flexibility:** choice between regression (`ols`) or naive (`naive`) forecast.
* **Portability:** runs on any machine with Python 3.10+.

## 🔮 Roadmap

* Add evaluation metrics (MAE, RMSE, MAPE).
* Compare with time series baselines (moving averages, seasonal naive).
* Build scikit-learn pipelines for richer preprocessing.
* Expand notebook for interactive teaching/demo.

## 📄 License

This project is under the [MIT License](LICENSE).