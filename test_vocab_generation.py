from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf
import data_utils

data_utils.create_vocabulary('dummy/dummy_vocab.txt', 'data/data.txt', 400000)
data_utils.initialize_vocabulary('dummy/dummy_vocab.txt')