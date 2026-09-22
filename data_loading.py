import kagglehub
import os
import pandas as pd

KAGGLE_ROUTES = {
    "car-traffic": "web-traffic-time-series-forecasting",
    "motor-temp": "wkirgsn/electric-motor-temperature",
    "wind-power": "theforcecoder/wind-power-forecasting",
    "store-sales": "rohitsahoo/sales-forecasting",
    "antarctica-climate": "sumanbera19/antarctica-climate-dataset-20062025-daily-data",
}

KAGGLE_DATASETS = ["motor-temp", "wind-power", "store-sales", "antarctica-climate", "traffic"]

KAGGLE_PATHS = {
    "motor-temp": "/Users/hoyle/.cache/kagglehub/datasets/wkirgsn/electric-motor-temperature/versions/3",
    "wind-power": "/Users/hoyle/.cache/kagglehub/datasets/theforcecoder/wind-power-forecasting/versions/2",
    "store-sales": "/Users/hoyle/.cache/kagglehub/datasets/rohitsahoo/sales-forecasting/versions/2/",
    "antarctica-climate": "/Users/hoyle/.cache/kagglehub/datasets/sumanbera19/antarctica-climate-dataset-20062025-daily-data/versions/1",
    "traffic": os.getcwd(),
}

KAGGLE_FILE_NAMES = {
    "motor-temp": "measures_v2.csv",
    "wind-power": "Turbine_Data.csv",
    "store-sales": "train.csv",
    "antarctica-climate": "antarctica_climate_2006_2025.csv",
    "traffic": "traffic.csv",
}

SERIES_META = {
    "motor-temp": {"freq_seconds": 0.5, "domain": "industrial sensor"},
    "wind-power": {"freq_seconds": 600, "domain": "renewable energy"},
    "store-sales": {"freq_seconds": 86400, "domain": "retail"},
    "antarctica-climate": {"freq_seconds": 86400, "domain": "environmental"},
    "traffic": {"freq_seconds": 3600, "domain": "transportation"},
}

def read_dataset(route):
    if route not in KAGGLE_ROUTES:
        print(f"invalid route: {route}")
        return

    identifier = KAGGLE_ROUTES[route]

    try:
        path = kagglehub.dataset_download(identifier)
    except Exception as e:
        print(f"failed to download {route} ({identifier}): {e}")
        return

    KAGGLE_PATHS[route] = path
    print(f"{route}: {path}")

"""
for route in KAGGLE_ROUTES:
    read_dataset(route)

print("\nFinal KAGGLE_PATHS:")
for k, v in KAGGLE_PATHS.items():
    print(f"  {k}: {v}")
"""

RAW = {}

def load_datasets():
    for i in KAGGLE_DATASETS:
        RAW[i] = pd.read_csv(os.path.join(KAGGLE_PATHS[i], KAGGLE_FILE_NAMES[i]))

SERIES = {}

def clean_datasets():
    load_datasets()

    # ---------- motor-temp ----------
    df = RAW["motor-temp"]

    chosen_profile = df["profile_id"].value_counts().idxmax()
    sub = df[df["profile_id"] == chosen_profile].reset_index(drop=True)
    target_col = "pm"
    SERIES["motor-temp"] = sub[target_col]

    # ---------- wind-power ----------
    df = RAW["wind-power"].rename(columns={"Unnamed: 0": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    target_col = "ActivePower"

    first_valid = df[target_col].first_valid_index()
    trimmed = df.loc[first_valid:]

    SERIES["wind-power"] = trimmed[target_col]

    # ---------- store-sales ----------
    df = RAW["store-sales"].copy()

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)  # confirm this against a few known rows
    daily_sales = df.groupby("Order Date")["Sales"].sum().sort_index()
    SERIES["store-sales"] = daily_sales

    # ---------- antarctica-climate ----------
    df = RAW["antarctica-climate"].copy()

    chosen_location = df["location"].unique()[0]
    sub = df[df["location"] == chosen_location].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.set_index("date").sort_index()
    SERIES["antarctica-climate"] = sub["temperature_2m_mean"]

    # ---------- traffic ----------
    df = RAW["traffic"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    SERIES["traffic"] = df["OT"]

    # ---------- store-sales: make the implicit gaps explicit ----------
    s = SERIES["store-sales"]
    full_range = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s_reindexed = s.reindex(full_range, fill_value=0.0)
    s_reindexed.index.name = "date"

    SERIES["store-sales"] = s_reindexed

    # ---------- wind-power: interpolate short gaps, then isolate longest clean run ----------
    s = SERIES["wind-power"]

    s_interp = s.interpolate(method="linear", limit=3)

    is_valid = s_interp.notna()
    run_id = (is_valid != is_valid.shift()).cumsum()
    run_lengths = s_interp[is_valid].groupby(run_id[is_valid]).size()
    longest_run_id = run_lengths.idxmax()

    clean_series = s_interp[(run_id == longest_run_id) & is_valid]

    SERIES["wind-power"] = clean_series


    return SERIES