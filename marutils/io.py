""" Read MAR outputs into xarray. """

import os
import numpy as np
from glob import glob
import re

import xarray as xr
import rioxarray

from marutils.georef import MAR_PROJECTION, MAR_BASE_PROJ4, create_crs

def _open_dataset(filename, projection=None, base_proj4=None, chunks=None, 
    errors='raise', **kwargs):
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
    xds = _reorganise_to_standard_cf(xds, errors=errors)
    # Apply chunking after dimensions have been renamed to standard names.
    if base_proj4 is not None and projection is not None:
        crs = create_crs(xds, projection, base_proj4)
        xds = _to_rio(xds, crs)
    if chunks is not None:
        xds = xds.chunk(chunks=chunks)
    return xds


def open_dataset(filenames, concat_dim='time', transform_func=None, chunks={'time': 366},
                 projection=MAR_PROJECTION, base_proj4=MAR_BASE_PROJ4,
                 errors='raise', **kwargs):
    """ Load single or multiple MAR NC files into a xr.Dataset.

    If multiple files are specified then they will be concatenated on the time axis.

    The following changes are applied to the dimensions to improve usability and 
    script portability between different MAR model runs:
    * 'X{n}_{n}'  --> x
    * 'Y{n}_{n}'  --> y
    * 'TIME'      --> 'time'
    * x & y km    --> metres

    You might also use indexing operations like .sel to subset datasets::
    
        mymar = open_dataset('MAR*.nc', dim='time', transform_func=lambda ds: ds.AL.sel(x=slice(.., ..)))

    Functionality based on http://xray.readthedocs.io/en/v0.7.1/io.html#combining-multiple-files
    See also http://xray.readthedocs.io/en/v0.7.1/dask.html

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
    :param errors: 'raise' or 'ignore'. If 'raise', errors encountered while trying
    to rename dataset dimensions to standards will be raised, otherwise they will
    be ignored.
    :type errors: str

    :return: concatenated dataset
    :rtype: xr.Dataset

    """

    def process_one_path(path, load_geo):
        if load_geo:
            ds = _open_dataset(
                path, projection=projection, base_proj4=base_proj4,
                chunks=chunks, errors=errors, **kwargs)
        else:
            ds = _open_dataset(
                path, projection=None, base_proj4=None, chunks=chunks, 
                errors=errors, **kwargs)
            
        # transform_func should do some sort of selection or
        # aggregation
        if transform_func is not None:
            ds = transform_func(ds)
        # load all data from the transformed dataset, to ensure we can
        # use it after closing each original file
        return ds

    paths = sorted(glob(filenames))
    if len(paths) == 0:
        raise FileNotFoundError('File(s) not found! : %s' %filenames)
        
    load_geo = [False] * len(paths)
    load_geo[0] = True
    datasets = [process_one_path(p, load_geo) for p in paths]
    if len(datasets) > 1:
        combined = xr.concat(datasets, concat_dim)
        return combined
    else:
        return datasets[0]


def _xy_dims_to_standard_cf(xds, errors='raise'):
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
        try:
            X_dim = xds['X']
        except KeyError:
            if errors == 'raise':
                raise ValueError('No X dimension identified from dataset.')
    if Y_dim is None:
        try:
            Y_dim = xds['Y']
        except KeyError:
            if errors == 'raise':
                raise ValueError('No Y dimension identified from dataset.')

    xds = xds.rename({X_dim.string: 'x'})
    xds = xds.rename({Y_dim.string: 'y'})

    xds['x'] = xds['x'] * 1000
    xds['y'] = xds['y'] * 1000

    return xds


def _time_dim_to_standard_cf(xds, errors='raise'):
    """ Rename the TIME dimension to time """
    try:
        xds = xds.rename({'TIME': 'time'})
    except ValueError:
        if errors == 'raise':
            raise ValueError('Dimension `TIME` not found in this dataset')
    return xds


def _reorganise_to_standard_cf(xds, errors='raise'):
    """ Reorganise dimensions, attributes into standard netCDF names. """
    xds = _xy_dims_to_standard_cf(xds, errors=errors)
    xds = _time_dim_to_standard_cf(xds, errors=errors)

    return xds


def _to_rio(xds, cc):
    """ Apply CRS to Dataset through rioxarray functionality. """
    return xds.rio.write_crs(cc.to_string(), inplace=True)


