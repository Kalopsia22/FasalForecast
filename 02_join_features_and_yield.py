"""
Join GEE features to ICRISAT yields.
Usage: python 02_join_features_and_yield.py [--data-dir data] [--yield data/yield_long.csv]

Writes (in --data-dir):
  features_all.csv      every GAUL district-season (yield filled where matched) -> used by the app
  training_table.csv    rows with a known yield -> used for training
  unmatched_districts.csv

ICRISAT uses fixed 1990-era districts; GAUL 2015 has later splits. Unsplit districts
match by name; split ones land in unmatched_districts.csv and need a manual decision.
"""
import argparse, difflib, os, re
import pandas as pd
from common import CROPS, GAUL_STATE_ALIAS, STATE_FALLBACK


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", str(s).lower())).strip()


ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data")
ap.add_argument("--yield", dest="yld", default="data/yield_long.csv")
ap.add_argument("--cutoff", type=float, default=0.85, help="fuzzy match threshold")
a = ap.parse_args()

frames = []
for crop in CROPS:
    p = f"{a.data_dir}/gee_features_{crop}.csv"
    if os.path.exists(p):
        d = pd.read_csv(p)
        d["crop"] = crop
        frames.append(d)
if not frames:
    raise SystemExit(f"No gee_features_*.csv in {a.data_dir}. Run step 01 first.")
feat = pd.concat(frames, ignore_index=True)
feat["gstate"] = feat["ADM1_NAME"].replace(GAUL_STATE_ALIAS)
feat["gkey"] = feat["ADM2_NAME"].map(norm)

yld = pd.read_csv(a.yld)
yld["ikey"] = yld["district"].map(norm)

# ICRISAT district -> GAUL district (fuzzy, within state; Telangana may sit under AP)
mapping, unmatched = {}, []
for st, g in yld.groupby("state"):
    pool = feat[feat.gstate.isin(STATE_FALLBACK.get(st, [st]))][["gstate", "gkey"]].drop_duplicates()
    names = pool.gkey.tolist()
    for ik in g.ikey.unique():
        hit = difflib.get_close_matches(ik, names, n=1, cutoff=a.cutoff)
        if hit:
            mapping[(st, ik)] = (pool[pool.gkey == hit[0]].gstate.iloc[0], hit[0])
        else:
            unmatched.append((st, ik))
yld["gstate"] = [mapping.get((s, k), (None, None))[0] for s, k in zip(yld.state, yld.ikey)]
yld["gkey"] = [mapping.get((s, k), (None, None))[1] for s, k in zip(yld.state, yld.ikey)]
yld = yld.dropna(subset=["gkey"]).rename(columns={"state": "istate", "district": "idistrict"})

out = feat.merge(yld[["crop", "gstate", "gkey", "year", "istate", "idistrict",
                      "area_kha", "yield_kgha"]],
                 on=["crop", "gstate", "gkey", "year"], how="left")
out = out.drop_duplicates(["crop", "ADM2_CODE", "year"])
out["matched"] = out.idistrict.notna()
out["state"] = out.istate.fillna(out.gstate)
out["district"] = out.idistrict.fillna(out.ADM2_NAME)

lead = ["crop", "state", "district", "ADM2_CODE", "year", "matched", "area_kha", "yield_kgha"]
cols = lead + [c for c in out.columns if c[:5] in ("ndvi_", "rain_", "tmean")]
out = out[cols].sort_values(["crop", "state", "district", "year"])
out.to_csv(f"{a.data_dir}/features_all.csv", index=False)
train = out[out.yield_kgha.notna() & out.ndvi_s1.notna()]
train.to_csv(f"{a.data_dir}/training_table.csv", index=False)
pd.DataFrame(unmatched, columns=["state", "icrisat_district"]).drop_duplicates().to_csv(
    f"{a.data_dir}/unmatched_districts.csv", index=False)

print(f"features_all: {len(out)} rows | training rows: {len(train)}")
print(train.groupby("crop").agg(rows=("year", "size"), districts=("district", "nunique")))
print(f"Unmatched ICRISAT districts: {len(set(unmatched))} (see unmatched_districts.csv)")
