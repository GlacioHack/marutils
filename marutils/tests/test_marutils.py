"""
Test functions for marutils
"""
import os
import inspect
import pytest

import marutils
from marutils import masking
from marutils import xhourly

@pytest.fixture()
def path_data():

    BASE_PATH = os.path.join(os.path.dirname(__file__), 'data')
    path2data = {}
    path2data['fn_nc_good'] = os.path.join(BASE_PATH, 'MARv3.11.2-ERA5-15km-TT.2020.06.01-10.nc')
    path2data['fn_nc_good_2'] = os.path.join(BASE_PATH, 'MARv3.11.2-ERA5-15km-TT.2020.06.11-15.nc')
    path2data['fn_nc_multi'] = os.path.join(BASE_PATH, 'MARv3.11.2-ERA5-15km-TT.2020.06.*.nc')
    path2data['fn_nc_xhourly'] = os.path.join(BASE_PATH, 'MARv3.11.2-ERA5-15km-TTH.2020.06.01.nc')
    return path2data

class TestIO:

    def test_open_dataset(self, path_data):
        ds = marutils.open_dataset(path_data['fn_nc_good'])

    def test_open_mfdataset(self, path_data):
        ds = marutils.open_dataset(path_data['fn_nc_multi'])

    def test_open_mfdataset_transform(self, path_data):
        ds = marutils.open_dataset(path_data['fn_nc_multi'],
            transform_func=lambda x: x.TT.sel(time=slice('2020-06-02', '2020-06-03')))
        assert len(ds.time) == 2


class TestGeoRef:

    def test_proj4(self, path_data):
        ds = marutils.open_dataset(path_data['fn_nc_good'])
        assert ds.rio.crs.to_proj4() == '+proj=stere +lat_0=70.5 +lon_0=-40 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs=True'

    def test_mpl_extent(self, path_data):
        ds = marutils.open_dataset(path_data['fn_nc_good'])
        extent = marutils.georef.get_mpl_extent(ds) 
        assert extent == pytest.approx([-757500.0631578948,
                                       682500.0631578948,
                                       1492500.1257022473,
                                       -1192500.1257022473])

class TestMasking:

    def test_gris_mask(self, path_data):
        ds = marutils.open_dataset(path_data['fn_nc_good'])
        mask = masking.gris_mask(ds)


class TestXHourly:

    def test_xhourly_to_time(self, path_data):
        ds = marutils.open_dataset(path_data['fn_nc_xhourly'])
        xd = xhourly.xhourly_to_time(ds.TTH)


   