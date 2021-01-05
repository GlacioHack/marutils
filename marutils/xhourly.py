""" 
Functions for working with X-hourly variables.
"""

import pandas as pd
import datetime as dt
import numpy as np

HDIM = 'ATMXH'

def _get_Xhourly_start_end(Xhourly_da):
    """ Return start and end timestamps of an X-hourly DataArray

    Used for DataArrays containing TIME and ATMXH coordinates.

    Assumes data are HOURLY, sub-hourly data are not catered for.

    :param Xhourly_da: an X-hourly DataArray
    :type Xhourly_da: xr.DataArray

    :return: start (0), end (1) timestamps in datetime.datetime type, freq (3)
    :rtype: tuple

    """

    hrs_in_da = len(Xhourly_da[HDIM])
    if np.mod(24, hrs_in_da) > 0:
        raise NotImplementedError

    freq = 24 / hrs_in_da

    dt_start = pd.to_datetime(Xhourly_da.time.isel(time=0).values)
    dt_start = dt_start - dt.timedelta(hours=(dt_start.hour-freq))

    dt_end = pd.to_datetime(Xhourly_da.time.isel(time=-1).values)
    dt_end = dt_end - dt.timedelta(hours=dt_end.hour)
    dt_end = dt_end + dt.timedelta(hours=24)

    return (dt_start, dt_end, freq)


def xhourly_to_time(xda):
    """ 
    Squeeze X-hourly dimension out of a variable, yielding hourly time dimension.

    Used for DataArrays with coordinates (y, x, ATMXH, time).

    Example::

        import marutils
        from marutils import xhourly
        mar = marutils.open_dataset('MAR.data.nc')
        hourly_tt = xhourly.xhourly_to_time(mar.TTH)

    :param xda: a DataArray of MAR data containing ATMXH coordinate
    :type xda: xr.DataArray

    :return: DataArray with ATMXH dimension removed and hours on the time dimension.
    :rtype: xr.DataArray

    """

    dt_start, dt_end, freq = _get_Xhourly_start_end(xda)

    index = pd.date_range(start=dt_start, end=dt_end, freq='%sH' % freq)

    hourly_da = xda.stack(TIME_H=(HDIM, 'time'))
    hourly_da['TIME_H'] = index
    hourly_da = hourly_da.rename({'TIME_H': 'time'})

    return hourly_da