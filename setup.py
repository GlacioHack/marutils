from setuptools import setup
from os import path

FULLVERSION = '1.0.0'
VERSION = FULLVERSION


setup()



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