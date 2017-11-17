# -*- coding: utf-8 -*-
"""
Load MAR NetCDF grids into a georaster instance or into xarray.

Needed because the NetCDF grids which MAR outputs are not geo-referenced in 
any of the several ways that GDAL understands.

@author Andrew Tedstone (a.j.tedstone@bristol.ac.uk)
@date March 2016

"""

from osgeo import osr, gdal
import os
import numpy as np
try:
    import pyproj
except ImportError:
    import mpl_toolkits.basemap.pyproj as pyproj
import xarray as xr
import cartopy.crs
from glob import glob
import re

import georaster

###
# MAR Grid Definitions 
# Deprecated. Remains here for reference

# grids = {}

# # 5 km grid:
# # ftp://ftp.climato.be/fettweis/MARv3.5/Greenland/readme.txt
# # http://nsidc.org/data/docs/daac/nsidc0092_greenland_ice_thickness.gd.html
# # note that the gdal transform uses CORNERS, not CENTRES (which are e.g. -800000)
# grids['5km'] = {'spatial_ref': '+proj=sterea +lat_0=90 +lat_ts=71 +lon_0=-39 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
#                 'geo_transform': (-802500, 5000, 0, -597500, 0, -5000)}

# # 10, 20 and 25 km grids are all the same as one another
# grids['25km'] = {'spatial_ref': '+proj=sterea +lat_0=70.5 +lon_0=-40 +k=1 +datum=WGS84 +units=m',
#                  'geo_transform': (-787500, 25000, 0, 1537500, 0, -25000)}

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



def open_xr(filename, X_name='X', Y_name='Y', **kwargs):
    """
    Load MAR NetCDF, setting X and Y coordinate names to X_name and Y_name,
    and multiplying by 1000 to convert coordinates to metres.

    Use **kawrgs to specify 'chunk' parameter if desired.

    """

    ds = xr.open_dataset(filename, **kwargs)

    # At some point I could convert this function to searching for the X
    # and Y dimensions 
    # Can't work out the regex at the moment
    # coords = list(ds.coords)
    # r = re.compile('/^[X,Y]*')
    # newlist = filter(r.match, coords)


    # Xname = [s for s in coords if "X" in s]
    # Yname = [s for s in coords if "Y" in s]



    # 25 km grid
    if 'X10_69' in ds.coords:
        ds.rename({'X10_69':X_name}, inplace=True)
        ds.rename({'Y18_127':Y_name}, inplace=True)

    # 10 km grid
    elif 'X10_153' in ds.coords:
        ds.rename({'X10_153':X_name}, inplace=True)
        ds.rename({'Y21_288':Y_name}, inplace=True)

    # 15 km grid
    elif 'X20_196' in ds.coords:
        ds.rename({'X20_196':X_name}, inplace=True)
        ds.rename({'Y19_216':Y_name}, inplace=True)
        
    # 20 km grid
    elif 'X12_84' in ds.coords:
        ds.rename({'X12_84':X_name}, inplace=True)
        ds.rename({'Y21_155':Y_name}, inplace=True)

    # 7.5 km grid
    elif 'X12_203' in ds.coords:
        ds.rename({'X12_203':X_name}, inplace=True)
        ds.rename({'Y20_377':Y_name}, inplace=True)

    ds['X'] = ds['X'] * 1000
    ds['Y'] = ds['Y'] * 1000

    return ds



def open_mfxr(files, dim='TIME', transform_func=None):
    """

    Load multiple MAR files into a single xarray object, performing some
    aggregation first to make this computationally feasible.

    E.g. select a single datarray to examine. 

    # you might also use indexing operations like .sel to subset datasets
    comb = read_netcdfs('MAR*.nc', dim='TIME',
                 transform_func=lambda ds: ds.AL)

    Based on http://xray.readthedocs.io/en/v0.7.1/io.html#combining-multiple-files
    See also http://xray.readthedocs.io/en/v0.7.1/dask.html

    """

    def process_one_path(path):        
        ds = open_xr(path,chunks={'TIME':366})
        # transform_func should do some sort of selection or
        # aggregation
        if transform_func is not None:
            ds = transform_func(ds)
        # load all data from the transformed dataset, to ensure we can
        # use it after closing each original file
        return ds

    paths = sorted(glob(files))
    datasets = [process_one_path(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined



def get_extent(ds):
    """ Return extent of xarray dataset [xmin,xmax,ymin,ymax] (corners) """
    xmin = float(ds.X.min())
    xmax = float(ds.X.max())
    ymin = float(ds.Y.min())
    ymax = float(ds.Y.max())

    # MAR values are grid centres so we need to do adjust to grid corners
    xsize, ysize = self.get_pixel_size(ds)
    xsize2 = xsize / 2
    ysize2 = ysize / 2
    xmin -= xsize2
    ymax -= ysize2
    xmax += xsize2
    ymin += ysize2

    return (xmin,xmax,ymin,ymax)



def get_pixel_size(ds):
    """ Return pixel dimensions in metres (xpixel, ypixel) """
    xpx = float(ds.X[1] - ds.X[0])
    ypx = float(ds.Y[0] - ds.Y[1])
    return (xpx, ypx)



def create_mar_res(xarray_obj, grid_info, gdal_dtype, ret_xarray=False, interp_type=gdal.GRA_NearestNeighbour):
    """ Resample other res raster to dimensions and resolution of MAR grid

    :param xarray_obj: A 2-D DataArray to resample with dimensions Y and X
    :type xarray_obj: xarray.DataArray
    :param grid_info: dict containing key-value pairs for nx, ny, xmin, ymax, xres, yres
    :type grid_info: dict
    :param gdal_dtype: a GDAL data type
    :type gdal_dtype: int
    :param ret_xarray: if True, return xarray DataArray, if False return GeoRaster
    :type ret_xarray: bool
    :param interp_type: a GDAL interpolation type
    :type interp_type: int
    
    :returns: mask at MAR resolution
    :rtype: GeoRaster.SingleBandRaster, xarray.DataArray

    """

    # Convert to numpy array and squeeze the extra dimension away
    as_array = xarray_obj.values.squeeze()
    if gdal_dtype == gdal.GDT_Byte:
        as_array = np.where(((np.isnan(as_array)) | (as_array == 0)),0,1)

    source_trans = (xarray_obj.X.values[0], xarray_obj.X.diff(dim='X').values[0], 0,
                    xarray_obj.Y.values[-1], 0, xarray_obj.Y.diff(dim='Y').values[0] * -1 )
    source_mask = georaster.SingleBandRaster.from_array(as_array, source_trans, 
        grids['25km']['spatial_ref'],gdal_dtype=gdal_dtype)
    # Reproject
    mask_7km = source_mask.reproject(source_mask.srs, 
                    grid_info['nx'], grid_info['ny'],
                    grid_info['xmin'], grid_info['ymax'],
                    grid_info['xres'], grid_info['yres'],
                    dtype=gdal_dtype,
                    interp_type=interp_type, nodata=0)

    if ret_xarray:
        cx, cy = mask_7km.coordinates(latlon=True)
        coords = {'Y': cy[:,0], 'X': cx[0,:]}
        da = xr.DataArray(mask_7km.r, dims=['Y', 'X'], coords=coords)
        return da
    else:
        return mask_7km



def create_annual_mar_res(multi_annual_xarray, MAR_MSK, mar_kws, gdal_dtype, **kwargs):
    """ Create a DataArray of masks, with TIME dimension 
    
    :param multi_annual_xarray: xarray with TIME, Y, X dimensions - TIME is annual
    :type multi_annual_xarray: xarray.DataArray
    :param MAR_MSK: 
    :type MAR_MSK:
    :param gdal_dtype: a GDAL data type
    :type gdal_dtype: int

     """

    years = multi_annual_xarray['TIME.year'].values
    # Set up store for masks at MAR resolution
    store = np.zeros((len(years), MAR_MSK.shape[0], MAR_MSK.shape[1]))
    n = 0
    for year in years:

        # Convert to MAR resolution
        mar_mask = create_mar_res(multi_annual_xarray.sel(TIME=str(year)), mar_kws, gdal_dtype, **kwargs)

        # Save - discard georaster information as we're putting into multi-year
        # DataArray instead.
        store[n, :, :] = mar_mask.r
        n += 1

    # Convert to multi-annual DataArray
    masks = xr.DataArray(store, coords=[multi_annual_xarray.TIME, MAR_MSK.Y, MAR_MSK.X], dims=['TIME', 'Y', 'X'])
    return masks



def create_proj4(ds_fn=None, ds=None, proj='sterea', lat_0=90,
    base='+k=1 +datum=WGS84 +units=m', return_pyproj=True):
    """ Return proj4 string for dataset.

    Create proj4 string using combination of values determined from dataset
    and those which must be known in advance (projection, lat_0).

    :param ds_fn: filename string of MARdataset
    :type ds_fn: str
    :param ds: xarray representation of MAR dataset opened using mar_raster
    :type ds: xr.Dataset
    :param proj: Proj.4 projection
    :type proj: str
    :param lat_0: Latitude of origin (degrees)
    :type lat_0: float, int
    :param base: base Proj.4 string for MAR
    :type base: str
    :param return_pyproj: If True return pyproj.Proj object, otherwise string
    :type return_pyproj: bool

    :return: Proj.4 string or pyproj.Proj object
    :rtype: str, pyproj.Proj

    """
    
    if ds == None:
        ds = self.open_xr(ds_fn)

    lat_ts = np.round(float(ds.LAT.sel(X=0,Y=0, method='nearest').values), 1)
    lon_0 = np.round(float(ds.LON.sel(X=0,Y=0, method='nearest').values), 1)

    proj4 = '%s +lat_ts=%s +lon_0=%s' %(base, lat_ts, lon_0)

    if return_pyproj:
        return pyproj.Proj(proj4)
    else:
        return proj4



def create_transform(ds_fn=None, ds=None):
    """ Return GDAL GeoTransform for dataset.

    :param ds_fn: filename string of MARdataset
    :type ds_fn: str
    :param ds: xarray representation of MAR dataset opened using mar_raster
    :type ds: xr.Dataset

    :return: GeoTransform (top left x, w-e size, 0, top left y, 0, n-s size)
    :rtype: tuple

    """
   
    if ds == None:
        ds = self.open_xr(ds_fn)

    # geotransform suitable for GDAL (i.e. cell corner not centre)
    # [xmin,xmax,ymin,ymax]
    extent = self.get_extent(ds)
    xsize, ysize = self.get_pixel_size(ds)
    # (top left x, w-e cell size, 0, top left y, 0, n-s cell size (-ve))
    trans = (extent[0], xsize, 0, extent[3], 0, ysize)

    return trans



# This function does not fill well in this module but is retained commented-out
# for reference
# def cartopy_proj(grid):
#     """ Return Cartopy CRS for specified MAR grid """

#     p = spatial_ref(grid)
#     crs = cartopy.crs.Stereographic(central_latitude=p.GetProjParm('latitude_of_origin'),
#         central_longitude=p.GetProjParm('central_meridian'))

#     return crs
