import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
nflx = pd.read_csv("HistoricalData_1768346229015.csv")
sp = pd.read_csv("SP500.csv")

# Clean Netflix data
nflx["Date"] = pd.to_datetime(nflx["Date"], format="%m/%d/%Y")
nflx["Close/Last"] = (
    nflx["Close/Last"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)
nflx = nflx.sort_values("Date")

# Clean S&P 500 data
sp["Date"] = pd.to_datetime(sp["observation_date"])
sp["SP500"] = pd.to_numeric(sp["SP500"], errors="coerce")
sp = sp[["Date", "SP500"]].sort_values("Date")

# Merge on common dates
df = pd.merge(nflx, sp, on="Date", how="inner")

# Normalise prices to compare relative movement
df["NFLX_norm"] = df["Close/Last"] / df["Close/Last"].iloc[0]
df["SP500_norm"] = df["SP500"] / df["SP500"].iloc[0]

# Plot comparison
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["NFLX_norm"], label="Netflix")
plt.plot(df["Date"], df["SP500_norm"], label="S&P 500")
plt.title("Netflix vs S&P 500 (Normalised Prices)")
plt.xlabel("Date")
plt.ylabel("Normalised Price")
plt.legend()
plt.tight_layout()
plt.show()