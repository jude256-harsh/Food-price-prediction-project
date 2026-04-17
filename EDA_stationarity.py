import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# ==============================
# LOAD CLEAN DATA
# ==============================
df = pd.read_csv("clean_food_price.csv")
df['date'] = pd.to_datetime(df['date'])

# Commodities to analyze
commodities = ['maize (white)', 'sorghum', 'beans']

# ==============================
# FUNCTION: ADF TEST
# ==============================
def adf_test(series, title=''):
    print(f"\nADF Test: {title}")
    result = adfuller(series)

    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")

    if result[1] <= 0.05:
        print("✅ Data is STATIONARY")
    else:
        print("❌ Data is NOT STATIONARY")

# ==============================
# LOOP THROUGH COMMODITIES
# ==============================
for commodity in commodities:

    print("\n" + "="*50)
    print(f"Analyzing: {commodity}")
    print("="*50)

    # Filter commodity
    data = df[df['commodity'] == commodity].copy()

    # Set index
    data = data.set_index('date').sort_index()

    ts = data['price']

    # ==============================
    # 1. PLOT ORIGINAL SERIES
    # ==============================
    plt.figure(figsize=(10, 4))
    plt.plot(ts, label='Original')
    plt.title(f"{commodity} Price Trend")
    plt.legend()
    plt.show()

    # ==============================
    # 2. ADF TEST (ORIGINAL)
    # ==============================
    adf_test(ts, f"{commodity} (Original)")

    # ==============================
    # 3. DIFFERENCING
    # ==============================
    ts_diff = ts.diff().dropna()

    # ==============================
    # 4. PLOT DIFFERENCED SERIES
    # ==============================
    plt.figure(figsize=(10, 4))
    plt.plot(ts_diff, label='Differenced', color='orange')
    plt.title(f"{commodity} After Differencing")
    plt.legend()
    plt.show()

    # ==============================
    # 5. ADF TEST (DIFFERENCED)
    # ==============================
    adf_test(ts_diff, f"{commodity} (Differenced)")