.. MARUtils documentation master file, created by
   sphinx-quickstart on Mon Nov 16 12:13:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

marutils - Utilities for working with MAR RCM outputs
=====================================================

This package contains utilities/tools that make it easier to work with outputs of the regional climate model
"MAR" (Modèle Atmosphérique Régional, http://mar.cnrs.fr), which is commonly used in studies of the Greenland and Antarctic ice sheets as well as other land areas.

* Load outputs straight into an xarray Dataset with standard dimensions names `(time, y, x)`.
* Provide a path with a wildcard expression (`*`) to automatically load several years of MAR files at once, concatenating them along the time axis.
* Plays nicely with Dask - chunking options default to 365-366 days (but can be changed at will).
* Loaded Datasets are automatically "geo-aware" using the rioxarray `.rio` accessor - no more trying to work out MAR's geo-referencing manually!
* Helper functions for turning X-hourly variables into continuous time series.
* Helper functions for masking.


Quick Start
-----------

To open a MAR dataset:: 


	import marutils
	# Open a time series for the 21st century.
	mar_outputs = marutils.open_dataset('MARv3.11.2/MAR*20*.nc')
	print(mar_outputs.coords)

Yields::

	Coordinates:
	  * x            (x) float32 -760000.0 -740000.0 -720000.0 ... 660000.0 680000.0
	  * y            (y) float32 -1180000.0 -1160000.0 ... 1480000.0 1500000.0
	  * SECTOR       (SECTOR) float32 1.0 2.0
	  * SECTOR1_1    (SECTOR1_1) float32 1.0
	  * time         (time) datetime64[ns] 2019-01-01T12:00:00 ... 2019-12-31T12:...
	  * ATMLAY3_3    (ATMLAY3_3) float32 0.99974793
	    spatial_ref  int64 0

What is proj.4 string for the geo-referencing of this dataset?::

	print(mar_outputs.rio.crs.to_proj4())

Yields::

	+proj=stere +lat_0=70.5 +lon_0=-40 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs=True


To return an X-hourly variable with a single, continuous hourly time variable::

	import marutils.xhourly
	hourly_air_temperatures = marutils.xhourly.xhourly_to_time(mar_outputs.TTH)


Installation
------------

At the moment, manually::

	git clone <repository>
	cd <repository>
	pip install -e .

Soon to be available via pyPI.



.. toctree::
   :maxdepth: 1
   :caption: API

   api
   



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
