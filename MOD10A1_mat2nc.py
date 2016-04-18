# -*- coding: utf-8 -*-
# @Author: Andrew Tedstone
# @Date:   2016-03-16 12:15:54
# @Last Modified by:   Andrew Tedstone
# @Last Modified time: 2016-04-14 15:24:47

import numpy as np
from scipy import io
import pandas as pd
import xarray as xr
import glob
import os

path = os.environ['rdsf'] + '/tedstone/raw_data/MOD10A1_MARv3.2_grid/'

files = glob.glob(path + '*best*mat')

# Import an exemplar nc file for the MAR 25km GR grid.
mar_25km = xr.open_dataset(path + 'MARv3.5.2_ERA_2014.nc', decode_times=False)
# Get space coordinates
x = mar_25km.X10_69
y = mar_25km.Y18_127


for f in files:

    print(f)

    as_mat = io.loadmat(f)['MODISdata']

    # Find year, the variable has some odd matrix dimensions
    year = as_mat['year'][0, 0][0, 0]

    # Extract albedo values
    # matrix layout: (110 x 60 x ddd)
    al_mean = as_mat['average'][0, 0]
    al_median = as_mat['median'][0, 0]
    al_n = as_mat['n'][0, 0]
    al_number = as_mat['number'][0, 0]
    al_stdev = as_mat['stdev'][0, 0]

    # Create time coordinates
    times = pd.date_range(str(year) + '-01-01', periods=al_mean.shape[2])

    # Bundle coordinates
    coords = {'X10_69': x,
              'Y18_127': y,
              'time': times}

    # Add coordinate info to data
    data = {'mean': (['Y18_127', 'X10_69', 'time'], al_mean),
        'median': (['Y18_127', 'X10_69', 'time'], al_median),
        'n': (['Y18_127', 'X10_69', 'time'], al_n),
        'number': (['Y18_127', 'X10_69', 'time'], al_number),
        'stdev': (['Y18_127', 'X10_69', 'time'], al_stdev)}

    # Add to DataSet
    dataset = xr.Dataset(data, coords=coords)

    # Add attributes
    dataset['mean'].attrs['description'] = 'average MODIS value within each grid box'
    dataset['median'].attrs['description'] = 'median MODIS value within each grid box'
    dataset['n'].attrs['description'] = 'number of good MODIS values within each grid box'
    dataset['number'].attrs['description'] = 'number of MODIS grid boxes assigned to the MAR grid box'
    dataset['stdev'].attrs['description'] = 'standard deviation of MODIS data assigned to each grid box'

    dataset.attrs['quality'] = 'produced only with data flagged as "good quality" within the MODIS product'
    dataset.attrs['authors'] = 'Marco Tedesco, Patrick Alexander'
    dataset.attrs['original_file'] = f
    dataset.attrs['converted_by'] = 'Andrew Tedstone'

    # Save it
    dataset.to_netcdf(f[:-4] + '.nc')
    dataset = None