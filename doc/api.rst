.. _api:

API Reference
=============

Full information about MARUtils' functionality is provided on this page.

Input/Output
------------

.. automodule:: marutils.io
	:members:


Georeferencing
--------------

Recall that MAR output files opened by :meth:`marutils.io.open_dataset` are automatically 
given the `.rio` accessor (see https://corteva.github.io/rioxarray/stable/).

.. automodule:: marutils.georef
	:members:


X-hourly data helpers
---------------------

.. automodule:: marutils.xhourly
	:members:


Masking
-------

.. automodule:: marutils.masking
	:members: