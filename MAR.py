"""
Helper functions for MAR Regional Climate model outputs, centred around 
rioxarray (corteva.github.io/rioxarray) and xarray (xarray.pydata.org).

@author Andrew Tedstone (andrew.tedstone@unifr.ch)
@date March 2016, November 2020.

"""


import os
import numpy as np
from glob import glob
import re
import datetime as dt

import xarray as xr
import rioxarray
import pandas as pd
from rasterio.crs import CRS

MAR_PROJECTION = 'stere'
MAR_BASE_PROJ4 = '+k=1 +datum=WGS84 +units=m'



def _open(filename, chunks=None, **kwargs):
    """
    Load MAR NetCDF, setting X and Y coordinate names to X_name and Y_name,
    and multiplying by 1000 to convert coordinates to metres.

    These 'odd' X and Y names originate from the clipping of extent undertaken
    by the Concat ferret routine.

    Use **kawrgs to specify 'chunk' parameter if desired.

    :param filename: The file path to open
    :type filename: str

    :return: opened Dataset
    :rtype: xr.Dataset with rio attribute

    """

    xds = xr.open_dataset(filename, **kwargs)
    xds = _reorganise_to_standard_cf(xds)
    # Apply chunking after dimensions have been renamed to standard names.
    if chunks is not None:
        xds = xds.chunk(chunks=chunks)
    crs = create_crs(xds)
    return _to_rio(xds, crs)



def open(filenames, concat_dim='time', transform_func=None,
    chunks={'time':366}, **kwargs):
    """ Load single or multiple MAR NC files concatenated on time axis into xr.Dataset.

    # you might also use indexing operations like .sel to subset datasets
    comb = open_multiple('MAR*.nc', dim='TIME',
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
        ds = open_xr(path, chunks=chunks, **kwargs)
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



################################################################################
# Geo-referencing and CF conventions.

def create_crs(xds):
    """ Create a Coordinate Reference System object for the dataset. """
    return CRS.from_proj4(create_proj4(xds))


def _xy_dims_to_standard_cf(xds):
    """ Coerce the X and Y dimensions into CF standard (and into metres). """
    X_dim = Y_dim = None
    for coord in xds.coords:
        if X_dim is None:
            X_dim = re.match('X[0-9]*_[0-9]*', coord)
        if Y_dim is None:
            Y_dim = re.match('Y[0-9]*_[0-9]*', coord)

        if (X_dim is not None) and (Y_dim is not None):
            break
    
    if X_dim is None:
        raise ValueError('No X dimension identified from dataset.')
    if Y_dim is None:
        raise ValueError('No Y dimension identified from dataset.')

    xds = xds.rename({X_dim.string:'x'})
    xds = xds.rename({Y_dim.string:'y'})

    xds['x'] = xds['x'] * 1000
    xds['y'] = xds['y'] * 1000

    return xds


def _reorganise_to_standard_cf(xds):
    """ Reorganise dimensions, attributes into standard netCDF names. """
    xds = _xy_dims_to_standard_cf(xds)
    xds = xds.rename({'TIME':'time'})

    return xds


def _to_rio(xds, cc):
    """ Apply CRS to Dataset through rioxarray functionality. """
    return xds.rio.write_crs(cc.to_string(), inplace=True)


def get_mpl_extent(xds):
    """ Return an extent tuple in the format required by matplotlib. 

    :param xds: a MAR XDataset opened through MAR.open().
    :dtype xds: xr.Dataset
    :returns: (xmin,xmax,ymin,ymax)
    :rtype: tuple
    """
    bounds = xds.rio.bounds()
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    return extent


def create_mar_res(xarray_obj, grid_info, gdal_dtype, ret_xarray=False, interp_type='nearest'):
    """ Resample other res raster to dimensions and resolution of MAR grid

    :param xarray_obj: A 2-D DataArray to resample with dimensions Y and X
    :type xarray_obj: xarray.DataArray
    :param grid_info: dict containing key-value pairs for nx, ny, xmin, ymax, xres, yres
    :type grid_info: dict
    :param gdal_dtype
    :type gdal_dtype: int
    :param ret_xarray: if True, return xarray DataArray, if False return GeoRaster
    :type ret_xarray: bool
    :param interp_type: a GDAL interpolation type
    :type interp_type: int
    
    :returns: mask at MAR resolution
    :rtype: GeoRaster.SingleBandRaster, xarray.DataArray

    """

    raise NotImplementedError

    if interp_type == 'nearest':
        interp_type = gdal.GRA_NearestNeighbour

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

    raise NotImplementedError
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


def create_proj4(xds, proj=MAR_PROJECTION, base=MAR_BASE_PROJ4):
    """ Return proj4 string for dataset.

    Create proj4 string using combination of values determined from dataset
    and those which must be known in advance (projection).

    :param xds: xarray representation of MAR dataset opened using mar_raster
    :type xds: xr.Dataset
    :param proj: Proj.4 projection
    :type proj: str
    :param base: base Proj.4 string for MAR
    :type base: str

    :return: Proj.4 string object
    :rtype: str

    """

    lat_0 = np.round(float(xds.LAT.sel(x=0, y=0, method='nearest').values), 1)
    lon_0 = np.round(float(xds.LON.sel(x=0, y=0, method='nearest').values), 1)

    proj4 = '+proj=%s +lon_0=%s +lat_0=%s %s' %(proj, lon_0, lat_0, base)

    return proj4



################################################################################
# Mask-related.

def mask_for_gris(ds_fn=None, ds=None):
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

    raise NotImplementedError

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



################################################################################
# Sub-daily-outputs related.

def get_Xhourly_start_end(Xhourly_da):
    """ Return start and end timestamps of an X-hourly DataArray

    Used for DataArrays containing TIME and ATMXH coordinates.

    Assumes data are HOURLY, sub-hourly data are not catered for.

    :param Xhourly_da: an X-hourly DataArray
    :type Xhourly_da: xr.DataArray

    :return: start (0), end (1) timestamps in datetime.datetime type, freq (3)
    :rtype: tuple

    """

    raise NotImplementedError

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

    raise NotImplementedError

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

    raise NotImplementedError

    dt_start, dt_end, freq = get_Xhourly_start_end(Xhourly_da)

    index = pd.date_range(start=dt_start, end=dt_end, freq='%sH' %freq)
    series = Xhourly_da.to_pandas().stack()
    series.index = index

    return series
   

