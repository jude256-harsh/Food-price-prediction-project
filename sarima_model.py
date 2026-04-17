import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("final_clean_food_prices.csv")
df['date'] = pd.to_datetime(df['date'])

commodities = ['Beans', 'Maize (white)', 'Sorghum']

results_summary = []

# ==============================
# LOOP
# ==============================
for commodity in commodities:

    print("\n" + "="*60)
    print(f"📊 Processing: {commodity}")
    print("="*60)

    # ==============================
    # FILTER
    # ==============================
    data = df[df['commodity'] == commodity].copy()
    data = data.set_index('date').sort_index()

    ts = data['price'].interpolate().ffill().bfill()

    if len(ts) < 20:
        print("❌ Not enough data:", commodity)
        continue

    # ==============================
    # SPLIT
    # ==============================
    split = int(len(ts) * 0.8)
    train = ts.iloc[:split]
    test = ts.iloc[split:]

    if len(test) < 2:
        print("❌ Test too small:", commodity)
        continue

    # ==============================
    # AUTO ARIMA
    # ==============================
    print("🔍 Running Auto ARIMA...")

    auto_model = auto_arima(
        train,
        seasonal=True,
        m=12,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )

    print("Best Order:", auto_model.order)
    print("Best Seasonal Order:", auto_model.seasonal_order)

    # ==============================
    # MODEL
    # ==============================
    model = SARIMAX(
        train,
        order=auto_model.order,
        seasonal_order=auto_model.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = model.fit(disp=False)

    # ==============================
    # FORECAST (FIXED)
    # ==============================
    forecast_values = results.forecast(steps=len(test))

    forecast = pd.Series(
        forecast_values.values if hasattr(forecast_values, "values") else forecast_values,
        index=test.index
    )

    # ==============================
    # SAFE ALIGNMENT (IMPORTANT FIX)
    # ==============================
    forecast = forecast.replace([np.inf, -np.inf], np.nan)

    mask = (~forecast.isna()) & (~test.isna())

    test_clean = test[mask]
    forecast_clean = forecast[mask]

    if len(test_clean) == 0:
        print("❌ No valid comparison data:", commodity)
        continue

    # ==============================
    # METRICS
    # ==============================
    mae = mean_absolute_error(test_clean, forecast_clean)

    mape = np.mean(
        np.abs((test_clean - forecast_clean) / np.maximum(test_clean, 1e-5))
    ) * 100

    print(f"✅ MAE: {mae:.2f}")
    print(f"✅ MAPE: {mape:.2f}%")

    results_summary.append([commodity, mae, mape])

    # ==============================
    # PLOT
    # ==============================
    plt.figure(figsize=(12,5))
    plt.plot(train, label="Train")
    plt.plot(test_clean, label="Actual")
    plt.plot(forecast_clean, label="Forecast", linestyle="--")
    plt.title(f"{commodity} SARIMA Forecast")
    plt.legend()
    plt.show()

    # ==============================
    # RESIDUALS
    # ==============================
    residuals = test_clean - forecast_clean

    plt.figure(figsize=(10,4))
    plt.plot(residuals)
    plt.axhline(0, linestyle="--")
    plt.title(f"{commodity} Residuals")
    plt.show()

# ==============================
# FINAL TABLE
# ==============================
results_df = pd.DataFrame(results_summary, columns=["Commodity", "MAE", "MAPE"])

print("\n📊 FINAL MODEL PERFORMANCE:")
print(results_df)