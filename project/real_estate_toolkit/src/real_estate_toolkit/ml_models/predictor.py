from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import polars as pl

class HousePricePredictor:
    def __init__(self, train_data_path: str, test_data_path: str):
        self.train_data = pl.read_csv('real_estate_toolkit/files/train.csv', null_values=['NA'])
        self.test_data = pl.read_csv('real_estate_toolkit/files/test.csv', null_values=['NA'])
        self.model_pipeline = {}

    def clean_data(self) -> None:
        """
        Perform comprehensive data cleaning.
        """
        self.train_clean_data = (
            self.train_data
            .fill_null(strategy="forward") 
            .with_columns(
                [
                    pl.col(col).cast(pl.Categorical)
                    for col in self.train_data.select(pl.col(pl.Categorical)).columns
                ]
            )
        )

        self.test_clean_data = (
            self.test_data
            .fill_null(strategy="forward")
            .with_columns(
                [
                    pl.col(col).cast(pl.Categorical)
                    for col in self.test_data.select(pl.col(pl.Categorical)).columns
                ]
            )
        )

    def prepare_features(self, target_column: str = 'SalePrice', selected_predictors: List[str] = None):
        """
        Prepare the dataset for machine learning by preprocessing features.
        """
        if selected_predictors is None:
            selected_predictors = [
                'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF',
                'YearBuilt', 'LotArea', 'Neighborhood'
            ]

        X = self.train_data.select(selected_predictors).to_pandas()
        y = self.train_data[target_column].to_numpy()

        numeric_features = [col for col in selected_predictors if col not in ['Neighborhood']]
        categorical_features = ['Neighborhood']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test


    def train_baseline_models(self):
        """
        Train and evaluate baseline machine learning models.
        """
        X_train, X_test, y_train, y_test = self.prepare_features()

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor()
        }

        results = {}

        for model_name, model in models.items():
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                        ('model', model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            results[model_name] = {
                'MSE': mean_squared_error(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred),
                'MAPE': mean_absolute_percentage_error(y_test, y_pred)
            }

            self.model_pipeline[model_name] = pipeline

        return results

    def forecast_sales_price(self, model_type: str = 'Linear Regression'):
        """
        Use the trained model to forecast house prices on the test dataset.
        """
        if model_type not in self.model_pipeline:
            raise ValueError(f"Model '{model_type}' has not been trained.")

        model = self.model_pipeline[model_type]

        """Select features consistent with training"""
        selected_predictors = [
            'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
            '1stFlrSF', 'YearBuilt', 'LotArea', 'Neighborhood'
        ]

        X_test = self.test_clean_data.select(selected_predictors).to_pandas()


        predictions = model.predict(X_test)

        submission = pl.DataFrame({
            'Id': self.test_clean_data['Id'],
            'SalePrice': predictions
        })

        submission.write_csv('real_estate_toolkit/src/real_estate_toolkit/ml_models/outputs/submission.csv')