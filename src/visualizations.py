from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_visualizations(housing, output_dir="."):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = {}

    plt.figure(figsize=(8, 5))
    sns.histplot(housing["median_house_value"], bins=20, kde=True)
    plt.title("House Price Distribution")
    plt.xlabel("Median House Value")
    plt.ylabel("Frequency")
    price_path = output_dir / "house_price_distribution.png"
    plt.tight_layout()
    plt.savefig(price_path)
    plt.close()
    plots["price_distribution"] = str(price_path)

    plt.figure(figsize=(8, 5))
    sns.histplot(housing["median_income"], bins=20, kde=True)
    plt.title("Median Income Distribution")
    plt.xlabel("Median Income")
    plt.ylabel("Frequency")
    income_path = output_dir / "median_income_distribution.png"
    plt.tight_layout()
    plt.savefig(income_path)
    plt.close()
    plots["income_distribution"] = str(income_path)

    numeric_cols = housing.select_dtypes(include=["number"]).columns.tolist()
    corr = housing[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    correlation_path = output_dir / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(correlation_path)
    plt.close()
    plots["correlation_heatmap"] = str(correlation_path)

    if {"median_income", "median_house_value"}.issubset(housing.columns):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=housing, x="median_income", y="median_house_value", alpha=0.6)
        plt.title("Median Income vs House Price")
        plt.xlabel("Median Income")
        plt.ylabel("Median House Value")
        scatter_path = output_dir / "income_vs_price_scatter.png"
        plt.tight_layout()
        plt.savefig(scatter_path)
        plt.close()
        plots["scatter_plot"] = str(scatter_path)

    return plots
