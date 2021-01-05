""" MAR georeferencing helpers. """

import numpy as np
from rasterio.crs import CRS

MAR_PROJECTION = 'stere'
MAR_BASE_PROJ4 = '+k=1 +datum=WGS84 +units=m'

def create_crs(xds, projection=MAR_PROJECTION, base_proj4=MAR_BASE_PROJ4):
    """ Create a Coordinate Reference System object for the dataset. """
    return CRS.from_proj4(create_proj4(xds, projection, base_proj4))


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