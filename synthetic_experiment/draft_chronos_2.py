import pandas as pd  # requires: pip install 'pandas[pyarrow]'
from chronos import Chronos2Pipeline
import matplotlib.pyplot as plt
from data_sources import create_dataloader
from utils_plots import plot_forecast


# --- pipeline initialization ---
pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")


# --- data loading ---
# Load historical target values and past values of covariates
# context_df = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet")
# print("--- Context DataFrame: ---")
# print(context_df.head())

dataset, dataloader = create_dataloader(data_length=100, batch_size=1)
first_batch = next(iter(dataloader))
context_df = pd.DataFrame(
    {
        "id": 'A',
        "timestamp": pd.date_range(start='2020-01-01', periods=1000, freq='h'),
        "target": first_batch[0].numpy(),
    }
)
print("--- Context DataFrame: ---")
print(context_df.head())

# (Optional) Load future values of covariates
# test_df = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/test.parquet")
test_df = None
# future_df = test_df.drop(columns="target")
# print("--- Future Covariates DataFrame: ---")
# print(future_df.head())

# --- forecasting ---
pred_df = pipeline.predict_df(
    context_df,
    # future_df=future_df,
    prediction_length=1004,  # Number of steps to forecast
    quantile_levels=[0.1, 0.5, 0.9],  # Quantiles for probabilistic forecast
    id_column="id",  # Column identifying different time series
    timestamp_column="timestamp",  # Column with datetime information
    target="target",  # Column(s) with time series values to predict
)
print("--- Predictions DataFrame: ---")
print(pred_df.head())

# --- plotting ---
plot_forecast(
    context_df,
    pred_df,
    None,
    target_column="target",
    # timeseries_id="DE",
    timeseries_id="A",
    title_suffix="(with covariates)",
)