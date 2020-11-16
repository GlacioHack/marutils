# from .marutils import (open_dataset,
# 	mask_for_gris,
# 	Xhourly_to_time,
# 	create_proj4,
# 	get_mpl_extent,
# 	create_crs)

from marutils.io import open_dataset

try:
    from marutils.version import version as __version__
except ImportError:  # pragma: no cover
    raise ImportError('marutils is not properly installed. If you are '
                      'running from the source directory, please instead '
                      'create a new virtual environment (using conda or '
                      'virtualenv) and then install it in-place by running: '
                      'pip install -e .')