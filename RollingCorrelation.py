import pandas as pd 
import matplotlib.pyplot as plt 

SP_PATH = "SP500.csv" 
NFLX_PATH = "HistoricalData_1768346229015.csv" 
WINDOW = 60  # trading days 

def load_prices(path): 
    df = pd.read_csv(path) 

    # FIX: include observation_date as a valid date column 
    date_col = next((c for c in df.columns if c.lower() in ["date", "timestamp", "observation_date"]), None) 
    if date_col is None: 
        raise ValueError(f"No date column found in {path}. Columns: {df.columns.tolist()}") 

  

    df[date_col] = pd.to_datetime(df[date_col]) 
    df = df.sort_values(date_col).set_index(date_col) 

    candidates = ["SP500", "Close/Last", "Adj Close", "AdjClose", "Close", "close", "adj close", "close/last"] 
    price_col = next((c for c in candidates if c in df.columns), None) 

  

    if price_col is None: 
        num_cols = df.select_dtypes(include="number").columns.tolist() 

        if not num_cols: 
            raise ValueError(f"No numeric price column found in {path}. Columns: {df.columns.tolist()}") 

        price_col = num_cols[0] 

    s = df[price_col].copy() 

  

    # clean "$" strings (Netflix Close/Last) 
    if s.dtype == "object": 
        s = (s.astype(str) 

               .str.replace("$", "", regex=False) 
               .str.replace(",", "", regex=False)) 

        s = pd.to_numeric(s, errors="coerce") 

    return s.dropna() 

  

sp = load_prices(SP_PATH).rename("sp") 
nflx = load_prices(NFLX_PATH).rename("nflx") 
px = pd.concat([sp, nflx], axis=1, join="inner").dropna() 
rets = px.pct_change().dropna() 

rolling_corr = rets["sp"].rolling(WINDOW).corr(rets["nflx"]) 

plt.figure() 
plt.plot(rolling_corr.index, rolling_corr.values) 
plt.title(f"{WINDOW}-Day Rolling Correlation: S&P 500 vs Netflix Returns") 
plt.xlabel("Date") 
plt.ylabel("Correlation") 
plt.grid(True, alpha=0.3) 
plt.tight_layout() 
plt.show() 
 