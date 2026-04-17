df = pd.read_csv("clean_food_price.csv")

df['date'] = pd.to_datetime(df['date'])
df.head()


df.set_index('date', inplace=True)

# 4. INJECT FUEL DATA
fuel_prices = {
    2017: 3549, 
    2018: 4023, 
    2019: 4085, 
    2020: 3851, 
    2021: 4051
}

# Now df.index.year will work
df['fuel_price'] = df.index.year.map(fuel_prices)

print("\n--- Data Ready ---")
print(df[['commodity', 'price', 'fuel_price']].head())
commodities = ['beans', 'sorghum', 'maize (white)']

for item in commodities:
    print(f"\n--- Processing {item.upper()} ---")
    
    # --- 1. FILTER & CLEAN ---
    item_df = df[df['commodity'].str.strip().str.lower() == item].copy()
for item in commodities:
    # --- 1. CLEANING (Already in your code) ---
    item_df = df[df['commodity'] == item].copy()
    # ... (numeric conversion, resampling, interpolation) ...
    item_df = item_df.dropna(subset=['price'])

    # --- 2. THE SPLIT (INSERT HERE) ---
    train_size = int(len(item_df) * 0.8)
    train = item_df.iloc[:train_size]
    test = item_df.iloc[train_size:]

    # --- 3. SCALING (UPDATE TO USE 'train') ---
    scaler = StandardScaler()
    # Only fit the scaler on 'train' to satisfy machine learning rules
    train_scaled = scaler.fit_transform(train[['price', 'fuel_price']])
    test_scaled = scaler.transform(test[['price', 'fuel_price']])

    # --- 4. AUTO ARIMA (UPDATE TO USE 'train') ---
    auto_model = auto_arima(
        train['price'], 
        exogenous=train[['fuel_price']],
        # ... other parameters ...
    )
     Scale data
scaler = StandardScaler()
item_df[['price','fuel_price']] = scaler.fit_transform(item_df[['price','fuel_price']])

# Auto ARIMA to find best parameters
auto_model = auto_arima(
    item_df['price'],
    exogenous=item_df[['fuel_price']],
    seasonal=True,
    m=12,
    suppress_warnings=True
)
results = {}

#  Ensure date is correct ONCE (important fix)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
df = df.sort_index()

# Commodities list
commodities = ['beans', 'sorghum', 'maize (white)']

for item in commodities:
    print(f"\n--- Processing {item.upper()} ---")

    #  1. Filter data
    item_df = df[df['commodity'].str.strip().str.lower() == item].copy()

    if item_df.empty:
        print(f"Skipping {item}: no data after filtering")
        continue

    #  2. Convert numeric
    item_df['price'] = pd.to_numeric(item_df['price'], errors='coerce')
    item_df['fuel_price'] = pd.to_numeric(item_df['fuel_price'], errors='coerce')

    #  3. Resample time series
    item_df = item_df.resample('MS').mean(numeric_only=True)

    #  4. Fix missing values (IMPORTANT)
    item_df['price'] = item_df['price'].interpolate()
    item_df['fuel_price'] = item_df['fuel_price'].ffill()

    item_df = item_df.dropna(subset=['price'])

    if item_df.empty:
        print(f"Skipping {item}: empty after cleaning")
        continue

    #  5. Scale data (NO warnings version)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(item_df[['price', 'fuel_price']].values)
    item_df[['price', 'fuel_price']] = scaled_values

    #  6. Auto ARIMA tuning
    auto_model = auto_arima(
        item_df['price'],
        exogenous=item_df[['fuel_price']],
        seasonal=True,
        m=12,
        suppress_warnings=True,
        stepwise=True
    )

    print(f"Best order for {item}: {auto_model.order}")
    print(f"Best seasonal order: {auto_model.seasonal_order}")

    #  7. Train SARIMAX
    model = SARIMAX(
        item_df['price'],
        exog=item_df[['fuel_price']],
        order=auto_model.order,
        seasonal_order=auto_model.seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=True
    )

    model_fit = model.fit(method='lbfgs', maxiter=500, disp=False)

    #  8. Future fuel (2022 forecast)
    future_fuel = np.array([4051] * 6)

    future_df = pd.DataFrame({
        'price': np.zeros(6),
        'fuel_price': future_fuel
    })

    future_scaled = scaler.transform(future_df.values)
    future_exog = future_scaled[:, 1].reshape(-1, 1)

    #  9. Forecast
    forecast = model_fit.get_forecast(steps=6, exog=future_exog)

    forecast_scaled = forecast.predicted_mean

    #  10. Convert back to real values
    combined = np.column_stack([forecast_scaled, future_exog.flatten()])
    forecast_real = scaler.inverse_transform(combined)[:, 0]

    print(f"\n Forecast for {item}:")
    print(np.round(forecast_real, 2))

    #  Save results
    results[item] = forecast_real
    results_dict = {} 

for item in commodities:
    print(f"\n--- Processing {item.upper()} ---")

    # 1. Filter and Clean
    item_df = df[df['commodity'].str.strip().str.lower() == item].copy()
    item_df['price'] = pd.to_numeric(item_df['price'], errors='coerce')
    item_df['fuel_price'] = pd.to_numeric(item_df['fuel_price'], errors='coerce')
    item_df = item_df.resample('MS').mean(numeric_only=True)
    item_df['price'] = item_df['price'].interpolate()
    item_df['fuel_price'] = item_df['fuel_price'].ffill()
    item_df = item_df.dropna(subset=['price'])

    # 2. Train-Test Split 
    train_size = int(len(item_df) * 0.8)
    train = item_df.iloc[:train_size]
    test = item_df.iloc[train_size:]

    # 3. Scaling & Modeling
    scaler = StandardScaler()
    # Fit on training data
    train_scaled = scaler.fit_transform(train[['price', 'fuel_price']])
    
    # Identify the best parameters
    auto_model = auto_arima(
        train['price'],
        exogenous=train[['fuel_price']],
        seasonal=True, m=12, suppress_warnings=True
    )

    # Build and fit the SARIMAX model
    model = SARIMAX(
        train['price'],
        exog=train[['fuel_price']],
        order=auto_model.order,
        seasonal_order=auto_model.seasonal_order
    )
    model_fit = model.fit(disp=False)

    # 4. Generate Test Forecast
    # Use test fuel prices for the exogenous variable
    test_exog = scaler.transform(test[['price', 'fuel_price']])[:, 1].reshape(-1, 1)
    forecast = model_fit.get_forecast(steps=len(test), exog=test_exog)
    
    # Save results into the dictionary for evaluation
    results_dict[item] = {
        'actual': test['price'],
        'pred': forecast.predicted_mean
    }

    print(f"Results saved for {item}")
    for item in commodities:
    print(f"\n{'='*20} Processing SARIMAX (With Fuel) for {item.upper()} {'='*20}")

    # [DATA PREP]
    item_df = df[df['commodity'].str.strip().str.lower() == item.lower()].copy()
    item_df['price'] = pd.to_numeric(item_df['price'], errors='coerce')
    item_df = item_df.resample('MS').mean(numeric_only=True).interpolate().dropna()

    # [SPLIT] 
    train_size = int(len(item_df) * 0.8)
    train, test = item_df.iloc[:train_size], item_df.iloc[train_size:]

    # [SCALING]
    scaler_y = StandardScaler()
    scaler_x = StandardScaler()
    
    # Scale Price (Target)
    train_price_scaled = scaler_y.fit_transform(train[['price']])
    
    # Scale Fuel (Exogenous helper)
    train_fuel_scaled = scaler_x.fit_transform(train[['fuel_price']])
    test_fuel_scaled = scaler_x.transform(test[['fuel_price']])

    # [MODELING]
    auto_model = auto_arima(train_price_scaled, exog=train_fuel_scaled, 
                            seasonal=True, m=12, suppress_warnings=True)
    
    model = SARIMAX(train_price_scaled, 
                    exog=train_fuel_scaled,
                    order=auto_model.order, 
                    seasonal_order=auto_model.seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    
    # Using 'nm' method and maxiter for better convergence
    model_fit = model.fit(disp=False, maxiter=200, method='nm')

    # --- 1. TRAINING ACCURACY (In-Sample Fit) ---
    train_pred_scaled = np.asarray(model_fit.fittedvalues).reshape(-1, 1)
    train_real_pred = scaler_y.inverse_transform(train_pred_scaled).flatten()
    # Skip first 2 months for stabilization
    train_mape = mean_absolute_percentage_error(train['price'][2:], train_real_pred[2:])
    train_acc = max(0, (1 - train_mape) * 100)

    # --- 2. TESTING ACCURACY (Out-of-Sample Forecast) ---
    forecast = model_fit.get_forecast(steps=len(test), exog=test_fuel_scaled)
    test_pred_scaled = np.asarray(forecast.predicted_mean).reshape(-1, 1)
    test_real_pred = scaler_y.inverse_transform(test_pred_scaled).flatten()
    
    test_mape = mean_absolute_percentage_error(test['price'], test_real_pred)
    test_acc = max(0, (1 - test_mape) * 100)

    # [METRICS]
    mae = mean_absolute_error(test['price'], test_real_pred)
    bias = np.mean(test['price'] - test_real_pred)
    direction = "UNDERESTIMATING" if bias > 0 else "OVERESTIMATING"
    overfit_gap = train_acc - test_acc

    # [PRINT RESULTS]
    print(f" SARIMAX PERFORMANCE (With Fuel):")
    print(f" TRAIN ACCURACY: {train_acc:.2f}%")
    print(f" TEST ACCURACY:  {test_acc:.2f}%")
    print(f" OVERFIT GAP:    {overfit_gap:.2f}%")
    print(f" MAE: {mae:.2f} UGX")
    print(f" TEST MAPE: {round(test_mape * 100, 2)}%")
    print(f" BIAS: {abs(bias):.2f} UGX ({direction})")

    # [PLOTTING]
    plt.figure(figsize=(10, 4))
    plt.plot(train.index, train['price'], label='History (Actual)', color='gray', alpha=0.3)
    plt.plot(train.index, train_real_pred, label='Training Fit', color='blue', linestyle=':', alpha=0.6)
    plt.plot(test.index, test['price'], label='Actual Price (Test)', color='black', marker='o')
    plt.plot(test.index, test_real_pred, label='SARIMAX Prediction', color='red', linestyle='--', marker='x')
    
    plt.title(f"{item.upper()} SARIMAX: Train {round(train_acc,1)}% | Test {round(test_acc,1)}%")
    plt.ylabel("Price (UGX)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    for item in commodities:
    item_df = df[df['commodity'].str.strip().str.lower() == item].copy()

    item_df['price'] = pd.to_numeric(item_df['price'], errors='coerce')
    item_df['fuel_price'] = pd.to_numeric(item_df['fuel_price'], errors='coerce')

    # ✅ FIXED LINE (NO COMMA)
    item_df = item_df.resample('MS').mean(numeric_only=True)

    # ✅ Fix missing data instead of deleting everything
    item_df['price'] = item_df['price'].interpolate()
    item_df['fuel_price'] = item_df['fuel_price'].ffill()

    item_df = item_df.dropna(subset=['price'])

    print(f"{item} after fix:", item_df.shape)

    if item_df.empty:
        print(f"Skipping {item}: No valid data.")
        continue

    final_model = SARIMAX(
    item_df['price'],
    exog=item_df[['fuel_price']],
    order=(1,1,0),              # 🔥 simpler AR structure
    seasonal_order=(0,0,0,0),   # 🔥 removes seasonal instability completely
    enforce_stationarity=True,
    enforce_invertibility=True,
    initialization='approximate_diffuse'  # 🔥 key fix
    )

    final_fit = final_model.fit(disp=False)

    future_fuel = np.array([4051]*6).reshape(-1,1)

    forecast = final_fit.get_forecast(steps=6, exog=future_fuel)

    print(f"\nForecast for {item}:")
    print(forecast.predicted_mean)