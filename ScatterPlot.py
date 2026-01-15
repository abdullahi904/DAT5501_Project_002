import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

  # Load files 
nflx = pd.read_csv("HistoricalData_1768346229015.csv") 
sp = pd.read_csv("SP500.csv") 

# Convert dates 
nflx["Date"] = pd.to_datetime(nflx["Date"]) 
sp["observation_date"] = pd.to_datetime(sp["observation_date"]) 

# Set index 
nflx.set_index("Date", inplace=True) 
sp.set_index("observation_date", inplace=True) 

# Select prices 
nflx_price = nflx["Close/Last"].str.replace("$","").astype(float) 
sp_price = sp["SP500"] 

# Returns 
nflx_ret = nflx_price.pct_change() 
sp_ret = sp_price.pct_change() 

# Align 
data = pd.concat([sp_ret, nflx_ret], axis=1).dropna() 

  
x = data.iloc[:,0]   # SP500 returns 
y = data.iloc[:,1]   # Netflix returns 

# Best fit line 
m, b = np.polyfit(x, y, 1) 

# Plot 
plt.scatter(x, y) 
plt.plot(x, m*x + b) 
plt.title("S&P 500 vs Netflix Returns") 
plt.xlabel("S&P 500 return") 
plt.ylabel("Netflix return") 
plt.grid(True) 
plt.show() 

print(f"Best fit: NFLX = {m:.4f} * SP500 + {b:.4f}") 