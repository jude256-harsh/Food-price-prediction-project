import pandas as pd

# ==============================
# LOAD DATASET
# ==============================
file_path = r"C:\Users\JUDE PRO\Downloads\archive\wfp_food_prices_uga.csv"
df = pd.read_csv(file_path)

# ==============================
# CLEAN DATE
# ==============================
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# ==============================
# CLEAN PRICE
# ==============================
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])

# ==============================
# FILTER YEARS (2017–2021)
# ==============================
df = df[
    (df['date'].dt.year >= 2017) &
    (df['date'].dt.year <= 2021)
]

# ==============================
# KEEP IMPORTANT COLUMNS
# ==============================
df = df[['date', 'commodity', 'price']]

# ==============================
# FILTER ONLY SELECTED COMMODITIES
# ==============================
selected_commodities = ['Beans', 'Sorghum', 'Maize (white)']
df = df[df['commodity'].isin(selected_commodities)]

# ==============================
# SORT
# ==============================
df = df.sort_values(by='date')

# ==============================
# GROUP BY (date + commodity)
# ==============================
df = df.groupby(['date', 'commodity'])['price'].mean().reset_index()

# ==============================
# 🔥 FIX MISSING DATES PER COMMODITY
# ==============================
fixed_data = []

for commodity in selected_commodities:
    temp = df[df['commodity'] == commodity].copy()

    temp = temp.set_index('date')

    full_dates = pd.date_range(start='2017-01-15', end='2021-12-15', freq='MS') + pd.DateOffset(days=14)

    temp = temp.reindex(full_dates)

    temp['commodity'] = commodity

    temp['price'] = temp['price'].interpolate(method='linear')

    temp = temp.reset_index().rename(columns={'index': 'date'})

    fixed_data.append(temp)

# Combine all commodities
df = pd.concat(fixed_data)

# Sort final dataset
df = df.sort_values(['commodity', 'date'])

# ==============================
# 🔥 SPLIT DATA FOR SARIMA MODELING
# ==============================

# Beans
beans_df = df[df['commodity'] == 'Beans'].copy()
beans_df.set_index('date', inplace=True)
beans_df = beans_df[['price']]

# Maize (white)
maize_df = df[df['commodity'] == 'Maize (white)'].copy()
maize_df.set_index('date', inplace=True)
maize_df = maize_df[['price']]

# Sorghum
sorghum_df = df[df['commodity'] == 'Sorghum'].copy()
sorghum_df.set_index('date', inplace=True)
sorghum_df = sorghum_df[['price']]

# ==============================
# OUTPUT
# ==============================
print("✅ Beans Data:")
print(beans_df.head())

print("\n✅ Maize Data:")
print(maize_df.head())

print("\n✅ Sorghum Data:")
print(sorghum_df.head())

# ==============================
# SAVE FINAL DATASETS
# ==============================
beans_df.to_csv(r"C:\Users\JUDE PRO\Downloads\archive\beans_time_series.csv")
maize_df.to_csv(r"C:\Users\JUDE PRO\Downloads\archive\maize_time_series.csv")
sorghum_df.to_csv(r"C:\Users\JUDE PRO\Downloads\archive\sorghum_time_series.csv")

print("\n✅ All datasets saved successfully!")