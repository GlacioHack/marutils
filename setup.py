from setuptools import setup
from os import path

FULLVERSION = '1.0.0'
VERSION = FULLVERSION

setup(name='marutils',
      version=FULLVERSION,
      description='Utilities and tools for working with MAR regional climate model outputs through xarray',
      url='https://www.github.com/GlacioHack/marutils/',
      author='Andrew Tedstone',
      license='BSD-3',
      packages=['marutils'],
      install_requires=['rioxarray', 'xarray', 'pandas', 'rasterio'],
      #extras_require=[],
      scripts=[],
      zip_safe=False)

write_version = True


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = path.join(path.dirname(__file__), 'marutils',
                             'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()


if write_version:
    write_version_py()