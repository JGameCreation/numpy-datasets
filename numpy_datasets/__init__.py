#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import images, timeseries, utils, preprocess, loader

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
