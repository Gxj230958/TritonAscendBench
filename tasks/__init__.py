"""tasks/ package marker.

Phase A' rewrites task kernel.py files to import from
``tasks._frozen_shim`` (the standalone shim subpackage). For that
import to resolve, ``tasks/`` must be a Python package -- i.e. it
must have an ``__init__.py``. The bench runner already adds the
bench root to sys.path; this file makes the subpackage discoverable.

Individual task directories (``tasks/level<N>/L1_xx_yyy/``) deliberately
do NOT have ``__init__.py`` -- they are bench data, not Python
subpackages, and their kernel.py is loaded by file path, not by
``import tasks.level1.L1_xx_yyy.kernel``.
"""
