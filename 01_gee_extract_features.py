"""
Extract monthly NDVI / rainfall / temperature per district and crop season
from Google Earth Engine.

Setup:  pip install earthengine-api pandas
        earthengine authenticate
Run:    python 01_gee_extract_features.py --project YOUR_GCP_PROJECT --crop all
        python 01_gee_extract_features.py --project P --crop wheat --years 2025   # current season
Outputs data/gee_features_<crop>.csv. Re-running resumes: years already in the file are skipped.

Notes: months with no data yet (future months, or data-latency lag, esp. ERA5-Land
~2-3 months) come back empty; the models handle that via the `lead` setting.
"""
import argparse, os, time
import ee
import pandas as pd
from common import CROPS, N_MONTHS, STATES_GAUL, season_months

PROPS = ["ADM1_NAME", "ADM2_NAME", "ADM2_CODE"]
BANDS = [f"{v}_s{i}" for i in range(1, N_MONTHS + 1) for v in ("ndvi", "rain", "tmean")]


def districts():
    return (ee.FeatureCollection("FAO/GAUL/2015/level2")
            .filter(ee.Filter.eq("ADM0_NAME", "India"))
            .filter(ee.Filter.inList("ADM1_NAME", STATES_GAUL)).select(PROPS))


def cropland():
    # Static ESA WorldCover 2021 cropland mask (class 40): a simplification for all years.
    return ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").eq(40)


def monthly(col, name, how, scale=1.0, offset=0.0):
    """Monthly aggregate that yields a fully-masked band when the month has no images."""
    empty = ee.Image.constant(0).selfMask().rename(name)
    agg = col.mean() if how == "mean" else col.sum()
    img = agg.multiply(scale).add(offset).rename(name)
    return ee.Image(ee.Algorithms.If(col.size().gt(0), img, empty))


def season_image(crop, year, mask):
    modis = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI")
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    era = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").select("temperature_2m")
    bands = []
    for i, (off, m) in enumerate(season_months(crop), start=1):
        s = ee.Date.fromYMD(year + off, m, 1)
        e = s.advance(1, "month")
        bands += [monthly(modis.filterDate(s, e), f"ndvi_s{i}", "mean", 0.0001),
                  monthly(chirps.filterDate(s, e), f"rain_s{i}", "sum"),
                  monthly(era.filterDate(s, e), f"tmean_s{i}", "mean", 1.0, -273.15)]
    return ee.Image.cat(bands).updateMask(mask)


def extract(crop, year, dists, mask, retries=3):
    fc = season_image(crop, year, mask).reduceRegions(
        collection=dists, reducer=ee.Reducer.mean(), scale=250, tileScale=4)
    for n in range(retries):
        try:
            df = pd.DataFrame([f["properties"] for f in fc.getInfo()["features"]])
            df = df.reindex(columns=PROPS + BANDS)
            df["year"] = year
            return df
        except Exception as e:
            print(f"   attempt {n + 1} failed: {e}")
            time.sleep(10 * (n + 1))
    raise RuntimeError(f"giving up on {crop} {year}")


def parse_years(s):
    a, _, b = s.partition("-")
    return range(int(a), int(b or a) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--crop", default="all", help="|".join(CROPS) + "|all")
    ap.add_argument("--years", default="2000-2017", help="e.g. 2000-2017 or 2025")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--list-states", action="store_true")
    a = ap.parse_args()
    ee.Initialize(project=a.project)

    if a.list_states:
        print(ee.FeatureCollection("FAO/GAUL/2015/level2")
              .filter(ee.Filter.eq("ADM0_NAME", "India"))
              .aggregate_array("ADM1_NAME").distinct().sort().getInfo())
        return

    dists, mask = districts(), cropland()
    print("Districts:", dists.size().getInfo())
    os.makedirs(a.data_dir, exist_ok=True)
    for crop in (CROPS if a.crop == "all" else [a.crop]):
        path = f"{a.data_dir}/gee_features_{crop}.csv"
        have = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        done = set(have["year"]) if len(have) else set()
        frames = [have] if len(have) else []
        for y in parse_years(a.years):
            if y in done and y != max(parse_years(a.years)):
                continue          # always refresh the latest requested year
            print(f"{crop} season {y}")
            new = extract(crop, y, dists, mask)
            frames = [f[f.year != y] for f in frames] + [new]
            pd.concat(frames).to_csv(path, index=False)
        print("saved", path)


if __name__ == "__main__":
    main()
