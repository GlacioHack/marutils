# -*- coding: utf-8 -*-
"""
Load MAR NetCDF grids into into xarray or a GeoRaster instance.

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
import pandas as pd
import datetime as dt

try:
    import georaster
except ImportError:
    gr_avail = False

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
    """ Load a MAR NetCDF file with the specified grid type using GeoRaster.

    DEPRECATED

    Parameters:
        filename : string, the path to the file with the relevant sub-dataset 
            referencing, e.g. 
            'NETCDF:"MARv3.5.2-10km-monthly-ERA-Interim-1979.nc":AL'

        grid : string, corresponding to entry in grids dictionary.


    Returns:
        MultiBandRaster instance, over full domain, all bands loaded

    """

    if gr_avail:

        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromProj4(grids[grid]['spatial_ref'])
        rast = georaster.MultiBandRaster(filename, spatial_ref=spatial_ref,
                                         geo_transform=grids[grid]['geo_transform'])

        # Flip the raster bands to restore correct orientation
        for i in rast.bands:
            rast.r[:, :, i-1] = np.flipud(rast.r[:,:, i-1])

        return rast

    else:
        print('GeoRaster dependency not available.')
        raise ImportError



def open_xr(filename, X_name='X', Y_name='Y', **kwargs):
    """
    Load MAR NetCDF, setting X and Y coordinate names to X_name and Y_name,
    and multiplying by 1000 to convert coordinates to metres.

    These 'odd' X and Y names originate from the clipping of extent undertaken
    by the Concat ferret routine.

    Use **kawrgs to specify 'chunk' parameter if desired.

    :param filename: The file path to open
    :type filename: str
    :param X_name: Name of X coordinate to rename to
    :type X_name: str
    :param Y_name: Name of Y coordinate to rename to
    :type Y_name: str

    :return: opened Dataset
    :rtype: xr.Dataset

    """

    ds = xr.open_dataset(filename, **kwargs)

    Xds = Yds = None
    for coord in ds.coords:
        if Xds is None:
            Xds = re.match('X[0-9]*_[0-9]*', coord)
        if Yds is None:
            Yds = re.match('Y[0-9]*_[0-9]*', coord)

        if (Xds is not None) and (Yds is not None):
            break
    

    if Xds is None:
        print ('No X dimension identified')
        raise ValueError
    if Yds is None:
        print ('No Y dimension identified')
        raise ValueError


    ds = ds.rename({Xds.string:X_name})
    ds = ds.rename({Yds.string:Y_name})

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

    :param files: filesystem path to open, with wildcard expression (*)
    :type files: str
    :param dim: name of dimension on which concatenate
    :type dim: str
    :param transform_func: a function to use to reduce/aggregate data
    :type transform_func: function

    :return: concatenated dataset
    :rtype: xr.Dataset

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
    """ Return extent of xarray dataset [xmin,xmax,ymin,ymax] (corners) 

    :param ds: Data from which to determine extent
    :type ds: xr.DataArray or xr.Dataset

    :return: (Xmin, Xmax, Ymin, Ymax)
    :rtype: tuple

    """
    xmin = float(ds.X.min())
    xmax = float(ds.X.max())
    ymin = float(ds.Y.min())
    ymax = float(ds.Y.max())

    # MAR values are grid centres so we need to do adjust to grid corners
    xsize, ysize = get_pixel_size(ds)
    xsize2 = xsize / 2
    ysize2 = ysize / 2
    xmin -= xsize2
    ymax -= ysize2
    xmax += xsize2
    ymin += ysize2

    return (xmin,xmax,ymin,ymax)



def get_pixel_size(ds):
    """ Return pixel dimensions in metres (xpixel, ypixel)

    :param ds: Data from which to determine pixel size
    :type ds: xr.DataArray or xr.Dataset

    :return: X pixel dimension, Y pixel dimension
    :rtype: tuple

    """
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

    if gr_avail == False:
        print('GeoRaster dependency not available.')
        raise ImportError

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
    masks = xr.DataArray(store, 
        coords=[multi_annual_xarray.TIME, MAR_MSK.Y, MAR_MSK.X], 
        dims=['TIME', 'Y', 'X'])
    return masks



def create_proj4(ds_fn=None, ds=None, proj='stere',
    base='+k=1 +datum=WGS84 +units=m', return_pyproj=True):
    """ Return proj4 string for dataset.

    Create proj4 string using combination of values determined from dataset
    and those which must be known in advance (projection).

    :param ds_fn: filename string of MARdataset
    :type ds_fn: str
    :param ds: xarray representation of MAR dataset opened using mar_raster
    :type ds: xr.Dataset
    :param proj: Proj.4 projection
    :type proj: str
    :param base: base Proj.4 string for MAR
    :type base: str
    :param return_pyproj: If True return pyproj.Proj object, otherwise string
    :type return_pyproj: bool

    :return: Proj.4 string or pyproj.Proj object
    :rtype: str, pyproj.Proj

    """
    
    if ds is None:
        ds = open_xr(ds_fn)

    lat_0 = np.round(float(ds.LAT.sel(X=0,Y=0, method='nearest').values), 1)
    lon_0 = np.round(float(ds.LON.sel(X=0,Y=0, method='nearest').values), 1)

    proj4 = '+proj=%s +lon_0=%s +lat_0=%s %s' %(proj, lon_0, lat_0, base)

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
   
    if ds is None:
        ds = open_xr(ds_fn)

    # geotransform suitable for GDAL (i.e. cell corner not centre)
    # [xmin,xmax,ymin,ymax]
    extent = get_extent(ds)
    xsize, ysize = get_pixel_size(ds)
    # (top left x, w-e cell size, 0, top left y, 0, n-s cell size (-ve))
    trans = (extent[0], xsize, 0, extent[3], 0, ysize)

    return trans



def gris_mask(ds_fn=None, ds=None):
    """ Return xarray representation of GrIS mask processed according to 
    Xavier Fettweis' method (see email XF-->AT 5 April 2018)

    Ferret method:
    yes? LET msk_tmp1            = if ( lat[d=1]  GE 75   AND lon[d=1] LE -75 ) then  0           else 1
    yes? LET msk_tmp2a           = if ( lat[d=1]  GE 79.5 AND lon[d=1] LE -67 ) then  0           else msk_tmp1
    yes? LET msk_tmp2            = if ( lat[d=1]  GE 81.2 AND lon[d=1] LE -63 ) then  0           else msk_tmp2a
    yes? let km3 = 15*15/(1000*1000)
    yes? LET msk2 = IF ( msk[d=1]  ge 50 ) then (1*msk_tmp2)*msk/100 else 0
    yes? let RUsum=RU*msk2*km3
    yes? list RUsum[k=1,x=@sum,y=@sum,l=@sum] 

    :param ds_fn: filename string of MARdataset
    :type ds_fn: str
    :param ds: xarray representation of MAR dataset opened using mar_raster
    :type ds: xr.Dataset

    :return: MAR mask with XF GrIS-specific post-processing applied
    :rtype: xr.DataArray

    """

    if ds is None:
        ds = open_xr(ds_fn)

    blank = xr.DataArray(np.zeros((len(ds.Y),len(ds.X))), dims=['Y','X'], 
        coords={'Y':ds.Y, 'X':ds.X})
    msk_tmp1 = blank.where((ds.LAT >= 75) & (ds.LON <= -75), other=1)
    msk_tmp2a = blank.where((ds.LAT >= 79.5) & (ds.LON <= -67), other=msk_tmp1)
    msk_tmp2 = blank.where((ds.LAT >= 81.2) & (ds.LON <= -63), other=msk_tmp2a)

    #msk2 = (ds.MSK.where(ds.MSK >= 50) * msk_tmp2) / 100
    msk_here = ds.MSK.where(ds.MSK >= 50, other=0)
    msk2 = (1*msk_tmp2) * msk_here/100

    return msk2



def get_Xhourly_start_end(Xhourly_da):
    """ Return start and end timestamps of an X-hourly DataArray

    Used for DataArrays containing TIME and ATMXH coordinates.

    Assumes data are HOURLY, sub-hourly data are not catered for.

    :param Xhourly_da: an X-hourly DataArray
    :type Xhourly_da: xr.DataArray

    :return: start (0), end (1) timestamps in datetime.datetime type, freq (3)
    :rtype: tuple

    """

    hrs_in_da = len(Xhourly_da['ATMXH'])
    if np.mod(24, hrs_in_da) > 0:
        raise NotImplementedError
    
    freq = 24 / hrs_in_da
    
    dt_start = pd.to_datetime(Xhourly_da.TIME.isel(TIME=0).values)
    dt_start = dt_start - dt.timedelta(hours=(dt_start.hour-freq))

    dt_end = pd.to_datetime(Xhourly_da.TIME.isel(TIME=-1).values)
    dt_end = dt_end - dt.timedelta(hours=dt_end.hour)
    dt_end = dt_end + dt.timedelta(hours=24)

    return (dt_start, dt_end, freq)



def squeeze_Xhourly(Xhourly_da):
    """ 
    Squeeze X-hourly dimension out of variable, yielding hourly TIME dimension.

    Used for DataArrays with coordinates (Y, X, ATMXH, TIME).

    :param Xhourly_da: an X-hourly DataArray containing ATMXH coordinate
    :type Xhourly_da: xr.DataArray

    :return: DataArray with ATMXH dimension removed and hours on the TIME dimension.
    :rtype: xr.DataArray

    """

    dt_start, dt_end, freq = get_Xhourly_start_end(Xhourly_da)

    index = pd.date_range(start=dt_start, end=dt_end, freq='%sH' %freq)

    hourly_da = Xhourly_da.stack(TIME_H=('ATMXH', 'TIME'))
    hourly_da['TIME_H'] = index
    hourly_da = hourly_da.rename({'TIME_H':'TIME'})

    return hourly_da



def Xhourly_pt_to_series(Xhourly_da):
    """
    Generate pd.Series of data from a point with an X-hourly dimension.

    Used for a DataArray with coordinates (Y=1, X=1, ATMXH, TIME).
    
    Assumes data are HOURLY, sub-hourly data are not catered for.

    :param Xhourly_da: an X-hourly DataArray containing ATMXH coordinate
    :type Xhourly_da: xr.DataArray

    :return: DataArray with ATMXH dimension removed and hours on the TIME dimension.
    :rtype: xr.DataArray

    """

    dt_start, dt_end, freq = get_Xhourly_start_end(Xhourly_da)

    index = pd.date_range(start=dt_start, end=dt_end, freq='%sH' %freq)
    series = Xhourly_da.to_pandas().stack()
    series.index = index

    return series
   




# This function does not fill well in this module but is retained commented-out
# for reference
# def cartopy_proj(grid):
#     """ Return Cartopy CRS for specified MAR grid """

#     p = spatial_ref(grid)
#     crs = cartopy.crs.Stereographic(central_latitude=p.GetProjParm('latitude_of_origin'),
#         central_longitude=p.GetProjParm('central_meridian'))

#     return crs
