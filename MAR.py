"""
Helper functions for MAR Regional Climate model outputs, centred around 
rioxarray (corteva.github.io/rioxarray) and xarray (xarray.pydata.org).

Main usage:

    mar_timeseries = mar_tools.open_dataset('ICE.*.nc')

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


def _open_dataset(filename, projection, base_proj4, chunks=None, **kwargs):
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
    crs = create_crs(xds, projection, base_proj4)
    return _to_rio(xds, crs)


def open_dataset(filenames, concat_dim='time', transform_func=None, chunks={'time': 366},
                 projection=MAR_PROJECTION, base_proj4=MAR_BASE_PROJ4, **kwargs):
    """ Load single or multiple MAR NC files into a xr.Dataset.

    # you might also use indexing operations like .sel to subset datasets
    comb = open_multiple('MAR*.nc', dim='TIME',
                 transform_func=lambda ds: ds.AL)

    Based on http://xray.readthedocs.io/en/v0.7.1/io.html#combining-multiple-files
    See also http://xray.readthedocs.io/en/v0.7.1/dask.html

    If multiple files are specified then they will be concatenated on the time axis.

    The following changes are applied to the dimensions to improve usability and 
    script portability between different MAR model runs:
        'X{n}_{n}'  --> x
        'Y{n}_{n}'  --> y
        'TIME'      --> 'time'
        x & y km    --> metres

    :param files: filesystem path to open, optionally with wildcard expression (*)
    :type files: str
    :param dim: name of dimension on which concatenate (only if multiple files specified)
    :type dim: str or None
    :param transform_func: a function to use to reduce/aggregate data
    :type transform_func: function
    :param chunks: A dictionary specifying how to chunk the Dataset. Use dimension names `time`, `y`, `x`.
    :type chunks: dict or None
    :param projection: The proj.4 name of the projection of the MAR file/grid.
    :type projection: str
    :param base_proj4: The basic proj.4 parameters needed to georeference the file.
    :type base_proj4: str

    :return: concatenated dataset
    :rtype: xr.Dataset

    """

    def process_one_path(path):
        ds = _open_dataset(path, projection, base_proj4,
                           chunks=chunks, **kwargs)
        # transform_func should do some sort of selection or
        # aggregation
        if transform_func is not None:
            ds = transform_func(ds)
        # load all data from the transformed dataset, to ensure we can
        # use it after closing each original file
        return ds

    paths = sorted(glob(filenames))
    datasets = [process_one_path(p) for p in paths]
    combined = xr.concat(datasets, concat_dim)
    return combined


################################################################################
# Geo-referencing and CF conventions.

def create_crs(xds, projection=MAR_PROJECTION, base_proj4=MAR_BASE_PROJ4):
    """ Create a Coordinate Reference System object for the dataset. """
    return CRS.from_proj4(create_proj4(xds, projection, base_proj4))


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

    xds = xds.rename({X_dim.string: 'x'})
    xds = xds.rename({Y_dim.string: 'y'})

    xds['x'] = xds['x'] * 1000
    xds['y'] = xds['y'] * 1000

    return xds


def _reorganise_to_standard_cf(xds):
    """ Reorganise dimensions, attributes into standard netCDF names. """
    xds = _xy_dims_to_standard_cf(xds)
    xds = xds.rename({'TIME': 'time'})

    return xds


def _to_rio(xds, cc):
    """ Apply CRS to Dataset through rioxarray functionality. """
    return xds.rio.write_crs(cc.to_string(), inplace=True)


def get_mpl_extent(xds):
    """ Return an extent tuple in the format required by matplotlib. 

    :param xds: a MAR XDataset opened through MAR.open().
    :type xds: xr.Dataset
    :returns: (xmin,xmax,ymin,ymax)
    :rtype: tuple
    """
    bounds = xds.rio.bounds()
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    return extent


def create_proj4(xds, proj, base):
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

    proj4 = '+proj=%s +lon_0=%s +lat_0=%s %s' % (proj, lon_0, lat_0, base)

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

    if ds is None:
        ds = open_xr(ds_fn)

    blank = xr.DataArray(np.zeros((len(ds.y), len(ds.x))), dims=['y', 'x'],
                         coords={'y': ds.y, 'x': ds.x})
    msk_tmp1 = blank.where((ds.LAT >= 75) & (ds.LON <= -75), other=1)
    msk_tmp2a = blank.where((ds.LAT >= 79.5) & (ds.LON <= -67), other=msk_tmp1)
    msk_tmp2 = blank.where((ds.LAT >= 81.2) & (ds.LON <= -63), other=msk_tmp2a)

    #msk2 = (ds.MSK.where(ds.MSK >= 50) * msk_tmp2) / 100
    msk_here = ds.MSK.where(ds.MSK >= 50, other=0)
    msk2 = (1*msk_tmp2) * msk_here/100

    return msk2


################################################################################
# Sub-daily-outputs related.

def _get_Xhourly_start_end(Xhourly_da):
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

    dt_start = pd.to_datetime(Xhourly_da.time.isel(time=0).values)
    dt_start = dt_start - dt.timedelta(hours=(dt_start.hour-freq))

    dt_end = pd.to_datetime(Xhourly_da.time.isel(time=-1).values)
    dt_end = dt_end - dt.timedelta(hours=dt_end.hour)
    dt_end = dt_end + dt.timedelta(hours=24)

    return (dt_start, dt_end, freq)


def Xhourly_to_time(Xhourly_da):
    """ 
    Squeeze X-hourly dimension out of variable, yielding hourly time dimension.

    Used for DataArrays with coordinates (y, x, ATMXH, time).

    :param Xhourly_da: an X-hourly DataArray containing ATMXH coordinate
    :type Xhourly_da: xr.DataArray

    :return: DataArray with ATMXH dimension removed and hours on the time dimension.
    :rtype: xr.DataArray

    """

    dt_start, dt_end, freq = _get_Xhourly_start_end(Xhourly_da)

    index = pd.date_range(start=dt_start, end=dt_end, freq='%sH' % freq)

    hourly_da = Xhourly_da.stack(TIME_H=('ATMXH', 'time'))
    hourly_da['TIME_H'] = index
    hourly_da = hourly_da.rename({'TIME_H': 'time'})

    return hourly_da
