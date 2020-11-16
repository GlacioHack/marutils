# marutils - Utilities for working with MAR RCM outputs

This package contains utilities/tools that make it easier to open MAR RCM outputs.

* Load outputs straight into an xarray Dataset with standard dimensions names `(time, y, x)`.
* Provide a path with a wildcard expression (`*`) to automatically load several years of MAR files at once, concatenating them along the time axis.
* Plays nicely with Dask - chunking options default to 365-366 days (but can be changed at will).
* Loaded Datasets are automatically "geo-aware" using the rioxarray `.rio` accessor - no more trying to work out MAR's geo-referencing manually!
* Helper functions for turning X-hourly variables into continuous time series.
* Helper functions for masking.


## QuickStart

```python
import marutils
# Open a time series for the 21st century.
mar_outputs = marutils.open_dataset('MARv3.11.2/MAR*20*.nc')

# Check out the coordinates.
print(mar_outputs.coords)
Coordinates:
  * x            (x) float32 -760000.0 -740000.0 -720000.0 ... 660000.0 680000.0
  * y            (y) float32 -1180000.0 -1160000.0 ... 1480000.0 1500000.0
  * SECTOR       (SECTOR) float32 1.0 2.0
  * SECTOR1_1    (SECTOR1_1) float32 1.0
  * time         (time) datetime64[ns] 2019-01-01T12:00:00 ... 2019-12-31T12:...
  * ATMLAY3_3    (ATMLAY3_3) float32 0.99974793
    spatial_ref  int64 0

# What is proj.4 string for the geo-referencing of this dataset?
print(mar_outputs.rio.crs.to_proj4())
+proj=stere +lat_0=70.5 +lon_0=-40 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs=True
```

## Installation

At the moment, manually:

	git clone <repository>
	cd <repository>
	pip install -e .

Soon to be available via pyPI.


## Working with X-hourly outputs

To return an X-hourly variable with a single, continuous hourly time variable:

```python
hourly_air_temperatures = marutils.xhourly.Xhourly_to_time(mar_outputs.TTH)
```

## Useful notes

### Pixel corners vs centres

It's useful to remember that netCDF file geo-referencing treats X,Y coordinates as pixel (cell) centres. In contrast, the rasterio/GDAL data model treats X,Y coordinates as pixel upper-left corners.

In the QuickStart section above we loaded a 20 km resolution dataset. The minimum and maximum X coordinates are -760000 and 680000 m respectively. These correspond to the centres of the min/max pixels.

Through the `.rio` accessor we can check on the bounds/complete spatial extent of the file:

```python
# What is the centre of the top-left cell? (xmin, ymin, xmax, ymax)
mar_outputs.rio.bounds()
(-770000.0, 1510000.0, 690000.0, -1190000.0)
```

We can see that these bounds account for the 20 km pixel size of this dataset.


## Caveats

The geo-referencing capabilities have so far been tested only for Greenland datasets. If you have problems trying to load other domains, please post an issue on this repository.

