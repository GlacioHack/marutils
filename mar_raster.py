# -*- coding: utf-8 -*-
"""
Load MAR NetCDF grids into a georaster instance.

Needed because the NetCDF grids which MAR outputs are not geo-referenced in 
any of the several ways that GDAL understands.

@author Andrew Tedstone (a.j.tedstone@bristol.ac.uk)
@date March 2016

"""

from osgeo import osr
import os
import numpy as np

import georaster

### 
# MAR Grid Definitions

grids = {}

# 5 km grid:
# ftp://ftp.climato.be/fettweis/MARv3.5/Greenland/readme.txt
# http://nsidc.org/data/docs/daac/nsidc0092_greenland_ice_thickness.gd.html
grids['5km'] = {'spatial_ref' : '+proj=stere +lat_0=90 +lat_ts=71 +lon_0=-39 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
                'geo_transform' : (-800000, 5000, 0, -600000, 0, -5000)}

grids['25km'] = {'spatial_ref' : '+proj=stere +lat_0=70.5 +lon_0=-40 +k=1 +datum=WGS84 +units=m +nodefs',
                 'geo_transform' : (-787500,25000,0,1537500,0,-25000)}

###


# GDAL won't load datasets unless bottomup set to no
# as a result we need to flip the rasters once they are read in.
os.environ['GDAL_NETCDF_BOTTOMUP'] = 'NO'

def load(filename,grid):
    """ Load a MAR NetCDF file with the specified grid type.

    Parameters:
        filename : string, the path to the file with the relevant sub-dataset 
            referencing, e.g. 
            'NETCDF:"MARv3.5.2-10km-monthly-ERA-Interim-1979.nc":AL'

        grid : string, corresponding to entry in grids dictionary.


    Returns:
        MultiBandRaster instance, over full domain, all bands loaded

    """

    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromProj4(grids[grid]['spatial_ref'])
    rast = georaster.MultiBandRaster(filename, spatial_ref=spatial_ref,
                                     geo_transform=grids[grid]['geo_transform'])

    # Flip the raster bands to restore correct orientation
    for i in rast.bands:
        rast.r[:, :, i-1] = np.flipud(rast.r[:, :, i-1])

    return rast     



