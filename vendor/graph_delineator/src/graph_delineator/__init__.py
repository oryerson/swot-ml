# The reservoir-aware delineation is the standard entry point: it is a strict
# superset of the original delineate.py (same public API, plus reservoir
# insertion/splitting) and is the version used to build the Columbia 0.5km2
# subbasins + graph network that the swot-ml pipeline consumes.
#
# Importing the package gives you the reservoir-aware delineate_basins(), which
# accepts the extra 'reservoirs' and 'min_reservoir_area_km2' arguments. Passing
# reservoirs=None reproduces the gauge-only behaviour of the old delineate.py.
#
# The original gauge-only module is still importable explicitly if needed:
#     from graph_delineator import delineate as delineate_gauges_only
from .delineate_reservoirs_working import *
