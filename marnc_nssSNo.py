"""
For MAR*nc files - example of using nssSNo variable to index into variables
with the snolay dimension.
"""

# '/scratch/atlantis1/at15963/MARout/GRx/x10/2012'

import xarray as xr 

dso = xr.open_dataset('MAR.20120701.x10.SnOptP_melt.nc').load()
dsn = xr.open_dataset('MAR.20120701.x10.SNICAR_melt_1h.nc').load()


figure(),dso.SolExt.isel(snolay=dso.nssSNo.astype('int') +1).plot(col='time',col_wrap=4)
figure(),dsn.SolExt.isel(snolay=dsn.nssSNo.astype('int') +1).plot(col='time',col_wrap=4)