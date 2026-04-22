import rasterio
from rasterio.enums import Resampling
import sys

#TODO add resolution argument for upsampling (default 0.25)

if(len(sys.argv) < 4):
    print("Usage: python upsample_tif.py <res> <input_tif> <output_tif>")
    sys.exit()

res = float(sys.argv[1])
input_tif = sys.argv[2]
output_tif = sys.argv[3]

with rasterio.open(input_tif) as src:
    data = src.read(
        out_shape=(src.count, src.height * 1/res, src.width * 1/res),
        resampling=Resampling.bilinear
    )
    transform = src.transform * src.transform.scale(res, res)
    profile = src.profile.copy()
    profile.update(width=src.width * 1/res, height=src.height * 1/res, transform=transform)
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(data)