from setuptools import setup

setup(name='marutils',
      version='0.2',
      description='Utilities and tools for working with MAR regional climate model outputs through xarray',
      url='https://www.github.com/atedstone/marutils/',
      author='Andrew Tedstone',
      license='BSD-3',
      packages=['marutils'],
      install_requires=['rioxarray', 'xarray', 'pandas', 'rasterio'],
      extras_require=[],
      scripts=[],
      zip_safe=False)
