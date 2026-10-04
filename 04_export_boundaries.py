"""Export simplified district boundaries (GeoJSON) for the app map.
Usage: python 04_export_boundaries.py --project YOUR_GCP_PROJECT"""
import argparse, json
import ee
from common import STATES_GAUL


def rnd(c):
    return round(c, 3) if isinstance(c, float) else [rnd(x) for x in c]


ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True)
ap.add_argument("--data-dir", default="data")
a = ap.parse_args()
ee.Initialize(project=a.project)

gaul = (ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(ee.Filter.eq("ADM0_NAME", "India")))
feats = []
for st in STATES_GAUL:                       # per state to stay under request limits
    fc = gaul.filter(ee.Filter.eq("ADM1_NAME", st)).select(
        ["ADM1_NAME", "ADM2_NAME", "ADM2_CODE"])
    fc = fc.map(lambda f: f.simplify(2000))
    for f in fc.getInfo()["features"]:
        f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
        f["properties"]["ADM2_CODE"] = int(f["properties"]["ADM2_CODE"])
        feats.append(f)
    print(st, len(feats))
json.dump({"type": "FeatureCollection", "features": feats},
          open(f"{a.data_dir}/districts.geojson", "w"))
print("wrote districts.geojson")
