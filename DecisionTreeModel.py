import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# File paths for the two high-level datasets
NFLX_CSV = "HistoricalData_1768346229015.csv"
SP500_CSV = "SP500.csv"

# Load Netflix and S&P 500 datasets
nflx_raw = pd.read_csv(NFLX_CSV)
sp_raw = pd.read_csv(SP500_CSV)

# Convert Netflix date column to datetime
nflx = nflx_raw.copy()
nflx["Date"] = pd.to_datetime(nflx["Date"], format="%m/%d/%Y")

# Remove currency symbols and convert price columns to float
for col in ["Close/Last", "Open", "High", "Low"]:
    nflx[col] = (
        nflx[col]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

# Sort Netflix data by time
nflx = nflx.sort_values("Date")

# Convert S&P 500 date column and ensure numeric values
sp = sp_raw.copy()
sp["Date"] = pd.to_datetime(sp["observation_date"])
sp["SP500"] = pd.to_numeric(sp["SP500"], errors="coerce")
sp = sp[["Date", "SP500"]].sort_values("Date")

# Merge Netflix and S&P 500 datasets on common trading dates
df = pd.merge(nflx, sp, on="Date", how="inner").dropna().reset_index(drop=True)

# Calculate daily returns for Netflix and the S&P 500
df["nflx_ret"] = df["Close/Last"].pct_change()
df["sp_ret"] = df["SP500"].pct_change()

# Calculate rolling 10-day volatility as a measure of risk
df["nflx_vol_10"] = df["nflx_ret"].rolling(10).std()
df["sp_vol_10"] = df["sp_ret"].rolling(10).std()

# Create lagged features representing information available at time t
df["nflx_ret_lag1"] = df["nflx_ret"].shift(1)
df["sp_ret_lag1"] = df["sp_ret"].shift(1)
df["nflx_vol_10_lag1"] = df["nflx_vol_10"].shift(1)
df["sp_vol_10_lag1"] = df["sp_vol_10"].shift(1)

# Define the target variable: whether Netflix price increases the next day
df["nflx_ret_next"] = df["nflx_ret"].shift(-1)
df["target_up_next_day"] = (df["nflx_ret_next"] > 0).astype(int)

# Remove rows with missing values caused by rolling windows and shifts
df_model = df.dropna().copy()

# Select features and target variable for modelling
features = [
    "nflx_ret_lag1",
    "sp_ret_lag1",
    "nflx_vol_10_lag1",
    "sp_vol_10_lag1"
]

X = df_model[features]
y = df_model["target_up_next_day"]

# Split the data using a time-aware split (first 80% train, last 20% test)
split = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train an interpretable decision tree classifier
model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=25,
    random_state=42
)
model.fit(X_train, y_train)

# Generate predictions on the test set
y_pred = model.predict(X_test)

# Output classification metrics to evaluate model performance
print(classification_report(y_test, y_pred, digits=3))
print(confusion_matrix(y_test, y_pred))

# Print the decision rules in text form for interpretability
print(export_text(model, feature_names=features))

# Visualise the trained decision tree for inclusion in the report
plt.figure(figsize=(16, 8))
plot_tree(
    model,
    feature_names=features,
    class_names=["Down/Flat", "Up"],
    filled=True,
    rounded=True,
    impurity=False,
    proportion=True,
    fontsize=10
)
plt.title("Decision Tree: Predict Netflix Next-Day Direction")
plt.tight_layout()
plt.show()


 

 