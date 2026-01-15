import pandas as pd
import numpy as np
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

# Prices
nflx_price = nflx["Close/Last"].str.replace("$","").astype(float)
sp_price = sp["SP500"]

# Returns
nflx_ret = nflx_price.pct_change()
sp_ret = sp_price.pct_change()

# Align
data = pd.concat([sp_ret, nflx_ret], axis=1).dropna()
data.columns = ["SP_Return", "NFLX_Return"]

# Actual direction
data["Actual"] = np.where(data["NFLX_Return"] > 0, "Up", "Down")

# Predicted direction (market-based rule)
data["Predicted"] = np.where(data["SP_Return"] > 0, "Up", "Down")

# Confusion matrix counts
tp = ((data["Actual"]=="Up") & (data["Predicted"]=="Up")).sum()
tn = ((data["Actual"]=="Down") & (data["Predicted"]=="Down")).sum()
fp = ((data["Actual"]=="Down") & (data["Predicted"]=="Up")).sum()
fn = ((data["Actual"]=="Up") & (data["Predicted"]=="Down")).sum()

labels = ["True Up", "True Down", "False Up", "False Down"]
values = [tp, tn, fp, fn]

# Bar chart
plt.figure()
plt.bar(labels, values)
plt.title("Predicted vs Actual Netflix Direction")
plt.ylabel("Number of Days")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print("True Up:", tp)
print("True Down:", tn)
print("False Up:", fp)
print("False Down:", fn)