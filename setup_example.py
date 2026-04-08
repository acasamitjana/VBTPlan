from os.path import join, exists, dirname
from os import listdir, makedirs
import pdb

BIDS_DIR = ''
ALIGNED_DIR = ''
CROSS_ALIGNED_DIR = '/'
CTV_DIR = ''

SYNTHSEG_WEIGHTS = ''
RESULTS_DIR = ''

LABELS_DICT = {
    'fons': 0,
    'intesti': 1,
    'recte': 2,
    'bufeta': 3,
    'sigma': 4,
    'vagina': 5,
}

LABELS_DICT_EN = {
    'background': 0,
    'bowel': 1,
    'rectum': 2,
    'bladder': 3,
    'sigma': 4,
    'vagina': 5
}

LABELS_DICT_EN_REV = {
    0: 'background',
    1: 'bowel',
    2: 'rectum',
    3: 'bladder',
    4: 'sigma',
    5: 'vagina',
}
