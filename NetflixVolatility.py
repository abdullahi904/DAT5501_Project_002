from importlib.resources import path
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

NFLX_PATH = "HistoricalData_1768346229015.csv" 
WINDOW = 60  # trading days 
TRADING_DAYS = 252 

def load_prices(path): 
    df = pd.read_csv(path) 
    date_col = next((c for c in df.columns if c.lower() in ["date", "timestamp"]), None) 
    if date_col is None: 
        raise ValueError(f"No date column found in {path}. Columns: {df.columns.tolist()}") 

 
    df[date_col] = pd.to_datetime(df[date_col]) 
    df = df.sort_values(date_col).set_index(date_col) 

    candidates = ["Adj Close", "AdjClose", "Close", "Close/Last", "close", "adj close", "close/last"] 
    price_col = next((c for c in candidates if c in df.columns), None) 

    if price_col is None: 
        num_cols = df.select_dtypes(include="number").columns.tolist() 

        if not num_cols: 
            raise ValueError(f"No numeric price column found in {path}. Columns: {df.columns.tolist()}") 
        price_col = num_cols[0] 

 

    s = df[price_col].copy() 

    if s.dtype == "object": 

        s = (s.astype(str) 

               .str.replace("$", "", regex=False) 
               .str.replace(",", "", regex=False)) 

        s = pd.to_numeric(s, errors="coerce") 


    return s.dropna() 

 
nflx_price = load_prices(NFLX_PATH).rename("nflx_price") 
nflx_ret = nflx_price.pct_change().dropna() 

rolling_vol_ann = nflx_ret.rolling(WINDOW).std() * np.sqrt(TRADING_DAYS) 

plt.figure() 
plt.plot(rolling_vol_ann.index, rolling_vol_ann.values) 
plt.title(f"Netflix {WINDOW}-Day Rolling Annualized Volatility") 
plt.xlabel("Date") 
plt.ylabel("Annualized volatility") 
plt.grid(True, alpha=0.3) 
plt.tight_layout() 
plt.show() 