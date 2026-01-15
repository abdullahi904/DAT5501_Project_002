# ConfusionMatrixUnitTest.py
import textwrap
import subprocess
import sys
import time
from pathlib import Path
import tempfile


def test_script_runs_and_prints_confusion_counts(tmp_path: Path):
    # temp csvs
    (tmp_path / "HistoricalData_1768346229015.csv").write_text(textwrap.dedent("""\
        Date,Close/Last
        2020-01-01,$100.00
        2020-01-02,$110.00
        2020-01-03,$105.00
        2020-01-04,$115.00
        2020-01-05,$120.00
        2020-01-06,$118.00
    """))

    (tmp_path / "SP500.csv").write_text(textwrap.dedent("""\
        observation_date,SP500
        2020-01-01,3000.0
        2020-01-02,3100.0
        2020-01-03,3050.0
        2020-01-04,3150.0
        2020-01-05,3200.0
        2020-01-06,3190.0
    """))

    # write ConfusionMatrix.py (your script, plt.show commented)
    script_path = tmp_path / "ConfusionMatrix.py"
    script_path.write_text(textwrap.dedent("""\
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
        # plt.show()

        print("True Up:", tp)
        print("True Down:", tn)
        print("False Up:", fp)
        print("False Down:", fn)
    """))

    # run ConfusionMatrix.py
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    # basic checks
    assert result.returncode == 0, f"script returned code {result.returncode}"

    stdout = result.stdout.strip()
    lines = stdout.splitlines()

    # Make sure all four counts are printed
    labels = ["True Up:", "True Down:", "False Up:", "False Down:"]
    found = {label: None for label in labels}

    for line in lines:
        for label in labels:
            if line.startswith(label):
                # get the numeric part after the colon
                parts = line.split(":", 1)
                if len(parts) == 2:
                    val_str = parts[1].strip()
                    # confirm it parses as an int
                    found[label] = int(val_str)

    for label in labels:
        assert found[label] is not None, f"Did not find line for '{label}' in output"


if __name__ == "__main__":
    # simple runner with timing
    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_script_runs_and_prints_confusion_counts(Path(tmpdir))
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