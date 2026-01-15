# DecisionTreeModelUnitTest.py
import textwrap
import subprocess
import sys
import time
from pathlib import Path
import tempfile


def test_script_runs_and_prints_tree_info(tmp_path: Path):
    # temp csvs
    (tmp_path / "HistoricalData_1768346229015.csv").write_text(textwrap.dedent("""\
        Date,Close/Last,Open,High,Low
        01/01/2020,$100.00,$99.00,$101.00,$98.00
        01/02/2020,$101.00,$100.00,$102.00,$99.00
        01/03/2020,$102.00,$101.00,$103.00,$100.00
        01/04/2020,$103.00,$102.00,$104.00,$101.00
        01/05/2020,$104.00,$103.00,$105.00,$102.00
        01/06/2020,$105.00,$104.00,$106.00,$103.00
        01/07/2020,$106.00,$105.00,$107.00,$104.00
        01/08/2020,$107.00,$106.00,$108.00,$105.00
        01/09/2020,$108.00,$107.00,$109.00,$106.00
        01/10/2020,$109.00,$108.00,$110.00,$107.00
        01/11/2020,$110.00,$109.00,$111.00,$108.00
        01/12/2020,$111.00,$110.00,$112.00,$109.00
        01/13/2020,$112.00,$111.00,$113.00,$110.00
        01/14/2020,$113.00,$112.00,$114.00,$111.00
        01/15/2020,$114.00,$113.00,$115.00,$112.00
        01/16/2020,$115.00,$114.00,$116.00,$113.00
        01/17/2020,$116.00,$115.00,$117.00,$114.00
        01/18/2020,$117.00,$116.00,$118.00,$115.00
        01/19/2020,$118.00,$117.00,$119.00,$116.00
        01/20/2020,$119.00,$118.00,$120.00,$117.00
    """))

    (tmp_path / "SP500.csv").write_text(textwrap.dedent("""\
        observation_date,SP500
        2020-01-01,3000.0
        2020-01-02,3010.0
        2020-01-03,3020.0
        2020-01-04,3030.0
        2020-01-05,3040.0
        2020-01-06,3050.0
        2020-01-07,3060.0
        2020-01-08,3070.0
        2020-01-09,3080.0
        2020-01-10,3090.0
        2020-01-11,3100.0
        2020-01-12,3110.0
        2020-01-13,3120.0
        2020-01-14,3130.0
        2020-01-15,3140.0
        2020-01-16,3150.0
        2020-01-17,3160.0
        2020-01-18,3170.0
        2020-01-19,3180.0
        2020-01-20,3190.0
    """))

    # write DecisionTreeModel.py
    script_path = tmp_path / "DecisionTreeModel.py"
    script_path.write_text(textwrap.dedent("""\
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
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Generate predictions on the test set
        y_pred = model.predict(X_test)

        # Output classification metrics to evaluate model performance
        print(classification_report(y_test, y_pred, digits=3))
        print(confusion_matrix(y_test, y_pred))

        # Print the decision rules in text form for interpretability
        rules_text = export_text(model, feature_names=features)
        print(rules_text)

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
        # plt.show()
    """))

    # run DecisionTreeModel.py
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    # basic checks
    assert result.returncode == 0, f"script returned code {result.returncode}"

    stdout = result.stdout.strip()

    # check that classification_report style lines appear
    # e.g., 'precision', 'recall', 'f1-score' should be in the report
    assert "precision" in stdout
    assert "recall" in stdout
    assert "f1-score" in stdout

    # check that confusion matrix-style bracket appears
    assert "[" in stdout and "]" in stdout

    # check that tree rules text is printed (export_text uses 'class:' in output)
    assert "class:" in stdout


if __name__ == "__main__":
    # simple runner with timing
    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_script_runs_and_prints_tree_info(Path(tmpdir))
        elapsed = time.perf_counter() - start
        print(f"TEST PASSED in {elapsed:.4f} seconds")
    except AssertionError as e:
        elapsed = time.perf_counter() - start
        print(f"TEST FAILED in {elapsed:.4f} seconds")
        print("Reason:", e)
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"TEST ERROR in {elapsed:.4f} seconds")
        print("Message:", e)