# -*- coding: utf-8 -*-
from inspect import getsourcefile
from os.path import join

# add resources to path
import os.path as path
current_dir = path.dirname(path.abspath(getsourcefile(lambda: 0)))

RESOURCES_DIR = join(current_dir[:current_dir.rfind(path.sep)], '..', '4-Resources')
