"""One-shot demo builder: creates data_demo/ and models_demo/ (synthetic features), then you can
run `streamlit run app.py`. Needs the ICRISAT raw CSV only if data/yield_long.csv is missing.
Usage: python run_demo.py [--raw path/to/icrisat_district_level_raw.csv]"""
import argparse, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="../icrisat_district_level_raw.csv")
a = ap.parse_args()
py = sys.executable
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/yield_long.csv"):
    subprocess.check_call([py, "00_prepare_yield.py", "--raw", a.raw])
subprocess.check_call([py, "make_demo_data.py"])
subprocess.check_call([py, "02_join_features_and_yield.py", "--data-dir", "data_demo"])
subprocess.check_call([py, "03_train_models.py", "--data-dir", "data_demo",
                       "--model-dir", "models_demo", "--fast"])
print("\nDemo ready. Now run:  streamlit run app.py")
