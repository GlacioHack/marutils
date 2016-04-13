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
import mpl_toolkits.basemap.pyproj as pyproj
import xarray as xr
import cartopy.crs

import georaster

###
# MAR Grid Definitions

grids = {}

# 5 km grid:
# ftp://ftp.climato.be/fettweis/MARv3.5/Greenland/readme.txt
# http://nsidc.org/data/docs/daac/nsidc0092_greenland_ice_thickness.gd.html
grids['5km'] = {'spatial_ref': '+proj=sterea +lat_0=90 +lat_ts=71 +lon_0=-39 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
                'geo_transform': (-800000, 5000, 0, -600000, 0, -5000)}

# 10, 20 and 25 km grids are all the same as one another
grids['25km'] = {'spatial_ref': '+proj=sterea +lat_0=70.5 +lon_0=-40 +k=1 +datum=WGS84 +units=m',
                 'geo_transform': (-787500, 25000, 0, 1537500, 0, -25000)}

grids['20km'] = grids['25km']
grids['10km'] = grids['25km']

###

"""
N.b. the pyproj transform yields lat/lon corners coords for bottom left that 
are not quite the same as those in the LAT and LON MAR grids. Perhaps the MAR
grids are cell centres? -- doesn't seem to be the case, I tried adding on
12.5km and the lat/lons still didn't match precisely.
"""

# GDAL won't load datasets unless bottomup set to no
# as a result we need to flip the rasters once they are read in.
os.environ['GDAL_NETCDF_BOTTOMUP'] = 'NO'



def load(filename, grid):
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
        rast.r[:, :, i-1] = np.flipud(rast.r[:,:, i-1])

    return rast



def load_xr(filename,X_name='X',Y_name='Y'):
    """
    Load MAR NetCDF, setting X and Y coordinate names to X_name and Y_name,
    and multiplying by 1000 to convert coordinates to metres.

    """

    ds = xr.open_dataset(filename)

    # 25 km grid
    if 'X10_69' in ds.coords:
        ds.rename({'X10_69':X_name},inplace=True)
        ds.rename({'Y18_127':Y_name},inplace=True)

    # 10 km grid
    elif 'X10_153' in ds.coords:
        ds.rename({'X10_153':X_name},inplace=True)
        ds.rename({'Y21_288':Y_name},inplace=True)
        
    # 20 km grid
    elif 'X12_84' in ds.coords:
        ds.rename({'X12_84':X_name},inplace=True)
        ds.rename({'Y21_155':Y_name},inplace=True)

    ds['X'] = ds['X'] * 1000
    ds['Y'] = ds['Y'] * 1000

    return ds



def cartopy_proj(grid):
    """ Return Cartopy CRS for specified MAR grid """

    p = spatial_ref(grid)
    crs = cartopy.crs.Stereographic(central_latitude=p.GetProjParm('latitude_of_origin'),
        central_longitude=p.GetProjParm('central_meridian'))

    return crs



def spatial_ref(grid):
    """Passed str grid returns OSR Spatial Reference instance"""

    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromProj4(grids[grid]['spatial_ref'])

    return spatial_ref



def proj(grid):
    """ Passed str grid returns a pyproj instance """

    srs = spatial_ref(grid)
    return pyproj.Proj(srs.ExportToProj4())
