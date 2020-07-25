#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Required settings to import not installed R packages in Python 
"""

from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects import StrVector



def settings_rpy2():
    base = importr('base')
    utils = importr('utils')
    utils = rpackages.importr('utils')
    packnames = ('caret', 'FRESA.CAD')
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))

settings_rpy2()

