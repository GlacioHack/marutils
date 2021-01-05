"""
Functions for returning standardised masks, e.g. of ice areas.
"""

import xarray as xr
from marutils import io
import numpy as np

def gris_mask(xds):
    """ Return a Greenland Ice Sheet mask.

    This method returns a mask of the GrIS processed according to 
    Xavier Fettweis' method (as per email XF-->AT 5 April 2018)

    Ferret method::
    
        yes? LET msk_tmp1            = if ( lat[d=1]  GE 75   AND lon[d=1] LE -75 ) then  0           else 1
        yes? LET msk_tmp2a           = if ( lat[d=1]  GE 79.5 AND lon[d=1] LE -67 ) then  0           else msk_tmp1
        yes? LET msk_tmp2            = if ( lat[d=1]  GE 81.2 AND lon[d=1] LE -63 ) then  0           else msk_tmp2a
        yes? let km3 = 15*15/(1000*1000)
        yes? LET msk2 = IF ( msk[d=1]  ge 50 ) then (1*msk_tmp2)*msk/100 else 0
        yes? let RUsum=RU*msk2*km3
        yes? list RUsum[k=1,x=@sum,y=@sum,l=@sum] 

    :param xds: filename of MARdataset, or an opened xarray representation
    :type ds_fn: str or xr.Dataset

    :return: MAR mask with XF GrIS-specific post-processing applied. Fractional values from 0 to 1.
    :rtype: xr.DataArray

    """

    if isinstance(xds, str):
        xds = io._open_dataset(xds)

    blank = xr.DataArray(np.zeros((len(xds.y), len(xds.x))), dims=['y', 'x'],
                         coords={'y': xds.y, 'x': xds.x})
    msk_tmp1 = blank.where((xds.LAT >= 75) & (xds.LON <= -75), other=1)
    msk_tmp2a = blank.where((xds.LAT >= 79.5) & (xds.LON <= -67), other=msk_tmp1)
    msk_tmp2 = blank.where((xds.LAT >= 81.2) & (xds.LON <= -63), other=msk_tmp2a)

    #msk2 = (xds.MSK.where(xds.MSK >= 50) * msk_tmp2) / 100
    msk_here = xds.MSK.where(xds.MSK >= 50, other=0)
    msk2 = (1*msk_tmp2) * msk_here/100

    return msk2