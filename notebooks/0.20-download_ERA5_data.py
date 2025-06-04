#!/usr/bin/env python
# coding: utf-8

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from src.download import download_era5_parallel

download_era5_parallel("../data/raw")

# In[ ]:

