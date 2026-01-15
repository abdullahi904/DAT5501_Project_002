# NetflixVolatilityUnitTest.py
import textwrap
import subprocess
import sys
import time
from pathlib import Path
import tempfile


def test_script_runs_and_prints_rolling_vol(tmp_path: Path):
    # temp csv
    (tmp_path / "HistoricalData_1768346229015.csv").write_text(textwrap.dedent("""\
        Date,Close/Last
        2020-01-01,$100.00
        2020-01-02,$110.00
        2020-01-03,$121.00
        2020-01-04,$133.10
        2020-01-05,$146.41
        2020-01-06,$161.05
    """))

    script_path = tmp_path / "NetflixVolatility.py"
    script_path.write_text(textwrap.dedent("""\
        from importlib.resources import path
        import numpy as np 
        import pandas as pd 
        import matplotlib.pyplot as plt 

        NFLX_PATH = "HistoricalData_1768346229015.csv" 
        WINDOW = 3  # smaller window for tiny test data 
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

        # Plot (disabled for unit test)
        # plt.figure() 
        # plt.plot(rolling_vol_ann.index, rolling_vol_ann.values) 
        # plt.title(f"Netflix {WINDOW}-Day Rolling Annualized Volatility") 
        # plt.xlabel("Date") 
        # plt.ylabel("Annualized volatility") 
        # plt.grid(True, alpha=0.3) 
        # plt.tight_layout() 
        # plt.show()

        # Simple textual output so test can assert something
        last_vol = rolling_vol_ann.dropna().iloc[-1]
        print(f"Rolling annualized volatility (window={WINDOW}) last value: {last_vol:.4f}")
    """))

    # run NetflixVolatility.py
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    # basic checks
    assert result.returncode == 0, f"script returned code {result.returncode}"

    stdout = result.stdout.strip()
    vol_lines = [
        line for line in stdout.splitlines()
        if "Rolling annualized volatility" in line and "last value" in line
    ]
    assert vol_lines, "no rolling volatility output line found"

    # parse the last volatility value just to ensure it is numeric
    line = vol_lines[0]
    # line format: "Rolling annualized volatility (window=3) last value: 0.1234"
    after_colon = line.split(":", 1)[1]
    vol_str = after_colon.strip().split()[0]
    float(vol_str)


if __name__ == "__main__":
    # simple runner with timing
    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_script_runs_and_prints_rolling_vol(Path(tmpdir))
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