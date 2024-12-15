from typing import List, Dict
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import os
import pyarrow  #vscode asked to install pyarrow in order to fullfill the data cleaning and the feature preparation
import statsmodels  #vscode asked to install statsmodels in order to fullfill the feature preparation

class MarketAnalyzer:
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data from a CSV file.

        Args:
            data_path (str): Path to the Ames Housing dataset
        """
        self.real_estate_data = pl.read_csv(data_path, null_values=["NA"])
        self.real_estate_clean_data = None
        self.output_dir = "src/real_estate_toolkit/analytics/outputs/"
        os.makedirs(self.output_dir, exist_ok=True)


    def clean_data(self) -> None:
        """
        Perform comprehensive data cleaning.
        """
        self.real_estate_clean_data = (
            self.real_estate_data
            .fill_null(strategy="forward")
            .with_columns(
                [
                    pl.col(col).cast(pl.Categorical)
                    for col in self.real_estate_data.select(pl.col(pl.Categorical)).columns
                ]
            )
        )
        self.real_estate_clean_data = self.real_estate_clean_data.to_pandas()

    def generate_price_distribution_analysis(self) -> pl.DataFrame:
        """
        Analyze sale price distribution using clean data.
        """
        if self.real_estate_clean_data is None:
            raise ValueError("Data is not cleaned. Please run clean_data() first.")


        price_statistics = pl.from_pandas(self.real_estate_clean_data).select([
            pl.col("SalePrice").mean().alias("Mean Price"),
            pl.col("SalePrice").median().alias("Median Price"),
            pl.col("SalePrice").std().alias("Price Std Dev"),
            pl.col("SalePrice").min().alias("Min Price"),
            pl.col("SalePrice").max().alias("Max Price")
        ])

        fig = px.histogram(
            self.real_estate_clean_data,
            x="SalePrice",
            nbins=50,
            title="Sale Price Distribution",
            labels={"SalePrice": "Sale Price"},
        )
        fig.write_html(os.path.join(self.output_dir, "price_distribution.html"))

        return price_statistics

    def neighborhood_price_comparison(self) -> pl.DataFrame:
        """
        Create a boxplot comparing house prices across different neighborhoods.
        """
        if self.real_estate_clean_data is None:
            raise ValueError("Data is not cleaned. Please run clean_data() first.")

        neighborhood_stats = self.real_estate_clean_data.groupby("Neighborhood")["SalePrice"].agg([
            "mean", "median", "std"
        ])

        fig = px.box(
            self.real_estate_clean_data,
            x="Neighborhood",
            y="SalePrice",
            title="Neighborhood Price Comparison",
            labels={"SalePrice": "Sale Price", "Neighborhood": "Neighborhood"},
        )
        fig.update_layout(xaxis_tickangle=-45)
        fig.write_html(os.path.join(self.output_dir, "neighborhood_price_comparison.html"))

        return pl.from_pandas(neighborhood_stats)

    def feature_correlation_heatmap(self, variables: List[str]) -> None:
        """
        Generate a correlation heatmap for variables input.
        """
        if self.real_estate_clean_data is None:
            raise ValueError("Data is not cleaned. Please run clean_data() first.")

        self.real_estate_clean_data = pl.from_pandas(self.real_estate_clean_data)
        correlation_matrix = self.real_estate_clean_data.select(variables).to_pandas().corr()

        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            title="Feature Correlation Heatmap",
            labels=dict(color="Correlation"),
        )
        fig.write_html(os.path.join(self.output_dir, "correlation_heatmap.html"))

    def create_scatter_plots(self) -> Dict[str, go.Figure]:
        """
        Create scatter plots exploring relationships between key features.
        """
        if self.real_estate_clean_data is None:
            raise ValueError("Data is not cleaned. Please run clean_data() first.")

        scatter_plots = {}

        relationships = [
            ("LotArea", "SalePrice"),
            ("YearBuilt", "SalePrice"),
            ("OverallQual", "SalePrice")
        ]

        for x, y in relationships:
            fig = px.scatter(
                self.real_estate_clean_data.to_pandas(),
                x=x,
                y=y,
                title=f"{y} vs. {x}",
                trendline="ols",
                labels={x: x, y: y}
            )
            fig.write_html(os.path.join(self.output_dir, f"scatter_{y}_vs_{x}.html"))
            scatter_plots[f"{y}_vs_{x}"] = fig

        return scatter_plots