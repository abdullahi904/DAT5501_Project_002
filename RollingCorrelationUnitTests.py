# RollingCorrelationUnitTest.py
import textwrap
import subprocess
import sys
import time
from pathlib import Path
import tempfile


def test_script_runs_and_computes_rolling_corr(tmp_path: Path):
    # temp csvs
    (tmp_path / "HistoricalData_1768346229015.csv").write_text(textwrap.dedent("""\
        Date,Close/Last
        2020-01-01,$100.00
        2020-01-02,$110.00
        2020-01-03,$121.00
        2020-01-04,$133.10
        2020-01-05,$146.41
        2020-01-06,$161.05
    """))
    (tmp_path / "SP500.csv").write_text(textwrap.dedent("""\
        observation_date,SP500
        2020-01-01,3000.0
        2020-01-02,3300.0
        2020-01-03,3630.0
        2020-01-04,3993.0
        2020-01-05,4392.3
        2020-01-06,4831.53
    """))

    script_path = tmp_path / "RollingCorrelation.py"
    script_path.write_text(textwrap.dedent("""\
        import pandas as pd 
        import matplotlib.pyplot as plt 

        SP_PATH = "SP500.csv" 
        NFLX_PATH = "HistoricalData_1768346229015.csv" 
        WINDOW = 3  # smaller window for tiny test data 

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

        # Plot (disabled for unit test)
        # plt.figure() 
        # plt.plot(rolling_corr.index, rolling_corr.values) 
        # plt.title(f"{WINDOW}-Day Rolling Correlation: S&P 500 vs Netflix Returns") 
        # plt.xlabel("Date") 
        # plt.ylabel("Correlation") 
        # plt.grid(True, alpha=0.3) 
        # plt.tight_layout() 
        # plt.show()

        # Simple textual output so test can assert something
        last_corr = rolling_corr.dropna().iloc[-1]
        print(f"Rolling correlation (window={WINDOW}) last value: {last_corr:.4f}")
    """))

    # run RollingCorrelation.py
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    # basic checks
    assert result.returncode == 0, f"script returned code {result.returncode}"

    stdout = result.stdout.strip()
    corr_lines = [
        line for line in stdout.splitlines()
        if "Rolling correlation" in line and "last value" in line
    ]
    assert corr_lines, "no rolling correlation output line found"

    # parse the last correlation value just to ensure it is numeric
    line = corr_lines[0]
    # line format: "Rolling correlation (window=3) last value: 0.1234"
    after_colon = line.split(":", 1)[1]
    corr_str = after_colon.strip().split()[0]
    float(corr_str)


if __name__ == "__main__":
    # simple runner with timing
    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_script_runs_and_computes_rolling_corr(Path(tmpdir))
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