# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import os

from pathlib import Path

__version__ = "0.1.1"
TITLE = 'rater'
VERSION = __version__
COPYRIGHT = 'xuming624@qq.com'
AUTHOR = 'xuming624@qq.com'

pwd_path = os.path.abspath(os.path.dirname(__file__))

USER_DIR = Path.expanduser(Path('~')).joinpath('.rater')
if not USER_DIR.exists():
    USER_DIR.mkdir()
USER_DATA_DIR = USER_DIR.joinpath('datasets')
if not USER_DATA_DIR.exists():
    USER_DATA_DIR.mkdir()
movielens_1m_path = os.path.join(USER_DATA_DIR, 'ml-1m.zip')
movielens_1m_dir = os.path.join(USER_DATA_DIR, 'ml-1m')
movielens_100k_path = os.path.join(USER_DATA_DIR, 'ml-100k.zip')
movielens_100k_dir = os.path.join(USER_DATA_DIR, 'ml-100k')
