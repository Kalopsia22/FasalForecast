"""
Train yield models for every crop at every forecast lead (1-6 months of season data),
validate honestly, and save everything the app needs.

Usage: python 03_train_models.py [--data-dir data] [--model-dir models] [--fast]

Target = yield minus the district's historical mean yield (the model learns the
anomaly). Baseline = that district mean (so "beat the baseline" is meaningful).
  * Leave-one-year-out  : the realistic setting (forecast a new year for known districts)
  * Leave-one-state-out : stress test of generalisation to unseen states
                          (baseline there is the global mean, a much weaker baseline)
Uncertainty = 10th/90th percentile quantile models (nominal 80% interval).
"""
import argparse, json, os, time
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from common import CROPS, N_MONTHS, make_features, ndvi_cols

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data")
ap.add_argument("--model-dir", default="models")
ap.add_argument("--fast", action="store_true", help="fewer trees; for testing")
ap.add_argument("--leads", default="1,2,3,4,5,6")
a = ap.parse_args()
LEADS = [int(x) for x in a.leads.split(",")]
P = dict(n_estimators=100 if a.fast else 250, learning_rate=0.1 if a.fast else 0.05,
         max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
         tree_method="hist", n_jobs=-1, random_state=0)


def new_models(Xtr, ytr):
    mean = XGBRegressor(**P).fit(Xtr, ytr)
    q = XGBRegressor(objective="reg:quantileerror",
                     quantile_alpha=np.array([0.1, 0.9]), **P).fit(Xtr, ytr)
    return mean, q


def fold(train, test, Xtr, Xte, scheme):
    if scheme == "year":
        dm = train.groupby("district_key")["yield_kgha"].mean()
        btr = train.district_key.map(dm).values
        bte = test.district_key.map(dm).fillna(train.yield_kgha.mean()).values
    else:
        c = train.yield_kgha.mean()
        btr, bte = np.full(len(train), c), np.full(len(test), c)
    mean, q = new_models(Xtr, train.yield_kgha.values - btr)
    m, qq = mean.predict(Xte), q.predict(Xte)
    lo, hi = np.minimum(qq.min(1), m), np.maximum(qq.max(1), m)
    return m + bte, lo + bte, hi + bte, bte


def cv(d, X, scheme):
    col = "year" if scheme == "year" else "state"
    parts = []
    for g in sorted(d[col].unique()):
        te = (d[col] == g).values
        if te.all() or te.sum() == 0:
            continue
        p, lo, hi, b = fold(d[~te], d[te], X[~te], X[te], scheme)
        parts.append(pd.DataFrame({"idx": d.index[te], "pred": p, "lo": lo, "hi": hi, "base": b}))
    return pd.concat(parts).set_index("idx")


def metrics(y, p, b, lo, hi):
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(n=len(y), rmse=rmse(y - p), mae=float(np.mean(np.abs(y - p))), r2=float(r2),
                base_rmse=rmse(y - b), improvement_pct=100 * (1 - rmse(y - p) / rmse(y - b)),
                coverage80=float(np.mean((y >= lo) & (y <= hi))), width=float(np.mean(hi - lo)))


tt = pd.read_csv(f"{a.data_dir}/training_table.csv")
tt["district_key"] = tt.state + "|" + tt.district
os.makedirs(a.model_dir, exist_ok=True)
t0 = time.time()
met, cvs, imps, summary = [], [], [], {}

for crop in CROPS:
    sub = tt[tt.crop == crop]
    if len(sub) < 300:
        print(f"[{crop}] only {len(sub)} rows - skipped")
        continue
    bundle = {}
    for k in LEADS:
        d = sub.dropna(subset=ndvi_cols(k) + ["yield_kgha"]).reset_index(drop=True)
        X = make_features(d, k)
        y = d.yield_kgha.values
        schemes = ["year"] + (["state"] if k == max(LEADS) else [])
        for sch in schemes:
            r = cv(d, X, sch)
            dd = d.loc[r.index]
            met.append(dict(crop=crop, lead=k, scheme=sch,
                            **metrics(y[r.index], r.pred.values, r.base.values, r.lo.values, r.hi.values)))
            if sch == "year":
                cvs.append(pd.DataFrame({
                    "crop": crop, "lead": k, "state": dd.state.values, "district": dd.district.values,
                    "district_key": dd.district_key.values, "year": dd.year.values,
                    "actual": dd.yield_kgha.values, "pred": r.pred.values, "lo": r.lo.values,
                    "hi": r.hi.values, "base": r.base.values}))
        # final model on all data
        dm = d.groupby("district_key")["yield_kgha"].mean()
        mean, q = new_models(X, y - d.district_key.map(dm).values)
        bundle[k] = dict(mean=mean, q=q, features=list(X.columns), district_mean=dm.to_dict())
        sample = X.sample(min(2000, len(X)), random_state=0)
        c = mean.get_booster().predict(xgb.DMatrix(sample), pred_contribs=True)[:, :-1]
        share = np.abs(c).mean(0)
        for f, s in zip(X.columns, share / share.sum()):
            imps.append(dict(crop=crop, lead=k, feature=f, importance=float(s)))
        m = [x for x in met if x["crop"] == crop and x["lead"] == k and x["scheme"] == "year"][-1]
        print(f"[{crop}] lead {k}: n={m['n']} RMSE {m['rmse']:.0f} vs baseline {m['base_rmse']:.0f} "
              f"({m['improvement_pct']:+.1f}%) R2 {m['r2']:.2f} cov80 {m['coverage80']:.2f}")
    joblib.dump(bundle, f"{a.model_dir}/model_{crop}.joblib", compress=3)

pd.DataFrame(met).to_csv(f"{a.model_dir}/metrics.csv", index=False)
pd.concat(cvs).to_csv(f"{a.model_dir}/cv_predictions.csv", index=False)
pd.DataFrame(imps).to_csv(f"{a.model_dir}/importance.csv", index=False)
json.dump({"trained": time.strftime("%Y-%m-%d %H:%M"), "fast_mode": a.fast,
           "rows": int(len(tt)), "years": [int(tt.year.min()), int(tt.year.max())]},
          open(f"{a.model_dir}/meta.json", "w"))
print(f"done in {time.time() - t0:.0f}s -> {a.model_dir}/")
