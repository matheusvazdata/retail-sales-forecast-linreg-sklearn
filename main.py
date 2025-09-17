# main.py
# =============================================================================
# Retail Sales Forecast — Minimal, Reproducible Script
#
# O que este script faz:
#   1) Carrega um dataset pequeno (12 meses) diretamente no código.
#   2) Converte o nome do mês para número (1–12).
#   3) Treina uma Regressão Linear (scikit-learn) usando Jan–Nov.
#   4) Prevê a venda de Dezembro por dois métodos possíveis:
#        - "ols": usa o modelo de regressão (≈ 3356.36 neste dataset)
#        - "naive": repete o passo do último mês conhecido (≈ 3300.00)
#   5) Gera dois gráficos e salva em reports/figures/, com nomes de arquivo EM INGLÊS:
#        - histogram_sales.png
#        - scatter_month_vs_sales.png
#
# Como rodar:
#   python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
#   pip install pandas numpy scikit-learn matplotlib
#   python main.py                    # usa método padrão "ols"
#   python main.py --method naive     # usa método ingênuo (passo constante)
#
# Observação importante:
#   Não usamos plt.show() para evitar warnings em ambientes sem backend interativo.
#   Os gráficos são sempre salvos em disco na pasta reports/figures/.
# =============================================================================

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ===== Configurações padrão ===================================================
DEFAULT_OUTPUT_DIR = Path("reports/figures")
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_METHOD = "ols"  # altere para "naive" se preferir padrão ingênuo
# ==============================================================================


def load_data() -> pd.DataFrame:
    """Retorna DataFrame com as colunas: mes (str), vendas (int)."""
    sales_dict = {
        "mes": [
            "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
            "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
        ],
        "vendas": [2000, 2200, 2300, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300],
    }
    return pd.DataFrame(sales_dict)


def add_month_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona a coluna mes_num (1–12) mapeando nome do mês."""
    map_month = {
        "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4,
        "Maio": 5, "Junho": 6, "Julho": 7, "Agosto": 8,
        "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12,
    }
    out = df.copy()
    out["mes_num"] = out["mes"].map(map_month)
    return out


def train_ols(df: pd.DataFrame) -> LinearRegression:
    """Treina Regressão Linear (mes_num -> vendas) usando Jan–Nov (1..11)."""
    train = df[df["mes_num"] <= 11]
    X = train[["mes_num"]].values
    y = train["vendas"].values
    model = LinearRegression()
    model.fit(X, y)
    return model


def forecast_december_ols(model: LinearRegression) -> float:
    """Prevê Dezembro (mes_num=12) usando o modelo OLS."""
    return float(model.predict(np.array([[12]]))[0])


def forecast_december_naive(df: pd.DataFrame) -> float:
    """Prevê Dezembro repetindo o passo do último mês (Nov - Out)."""
    nov = df.loc[df["mes_num"] == 11, "vendas"].item()
    out = df.loc[df["mes_num"] == 10, "vendas"].item()
    step = nov - out
    return float(nov + step)  # para este dataset: 3200 + 100 = 3300


def plot_histogram(df: pd.DataFrame, outdir: Path) -> Path:
    """Gera histograma de vendas e salva como histogram_sales.png."""
    path = outdir / "histogram_sales.png"
    plt.figure()
    plt.hist(df["vendas"], bins=6)
    plt.title("Sales Histogram")
    plt.xlabel("Sales")
    plt.ylabel("Frequency")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_scatter(df: pd.DataFrame, model: LinearRegression, outdir: Path) -> Path:
    """Gera scatter Mês vs Vendas (com reta da regressão) e salva como scatter_month_vs_sales.png."""
    path = outdir / "scatter_month_vs_sales.png"
    plt.figure()
    # pontos observados
    plt.scatter(df["mes_num"], df["vendas"], label="Observed")
    # reta da regressão (1..12 para visualizar extrapolação)
    x_line = np.arange(1, 13).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, label="Linear Regression")
    plt.title("Month (1–12) vs Sales")
    plt.xlabel("Month (1–12)")
    plt.ylabel("Sales")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retail Sales Forecast — train OLS on Jan–Nov and forecast December. "
            "Also saves histogram and scatter plots under reports/figures/."
        )
    )
    parser.add_argument(
        "--method",
        choices=["ols", "naive"],
        default=DEFAULT_METHOD,
        help='Forecast method for December: "ols" (linear regression) or "naive" (constant step).',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output figures (default: reports/figures).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outdir: Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Dados
    df = load_data()
    df = add_month_numeric(df)

    # 2) Treino (Jan–Nov)
    model = train_ols(df)

    # 3) Previsão para Dezembro (método escolhido)
    if args.method == "naive":
        y_dec = forecast_december_naive(df)
        method_label = "naive (constant step)"
    else:
        y_dec = forecast_december_ols(model)
        method_label = "OLS (linear regression)"

    print(f"December forecast ({method_label}): {y_dec:.2f}")

    # 4) Gráficos (salvos em disco; sem plt.show())
    hist_path = plot_histogram(df, outdir)
    scat_path = plot_scatter(df, model, outdir)
    print("Figures saved to:")
    print(f" - {hist_path}")
    print(f" - {scat_path}")


if __name__ == "__main__":
    main()