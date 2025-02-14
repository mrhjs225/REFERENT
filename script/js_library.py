from tqdm import tqdm
from pprint import pprint
from collections import UserDict
from re import S
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pickle
import random
import shutil
import time
import re
import requests
import pickle
import copy
import sys


def violin_and_boxplot(target_list):
    print('len:{}, mean:{}'.format(len(target_list), np.mean(target_list)))
    plt.style.use('default')
    fig, ax = plt.subplots()
    ax.violinplot([target_list], vert=False)
    ax.boxplot([target_list], vert=False)
    # plt.xscale('log')
    plt.yticks([1], ['target'])
    plt.show()

def bar_graph(sequence_stat):
    x = np.arange(len(sequence_stat))
    sequence_str = sequence_stat.keys()
    values = sequence_stat.values()

    plt.bar(x, values)
    plt.xticks(x, sequence_str)
    plt.show()

def make_dirs(path):
   try:
        os.makedirs(path)
   except OSError:
       if not os.path.isdir(path):
           raise

def read_pickle(file_dir):
    with open(file_dir, 'rb') as f:
        return pickle.load(f)

def write_pickle(file_content, file_dir):
    with open(file_dir, 'wb') as f:
        pickle.dump(file_content, f)

def sort_dict_by_value(target_dict):
    return sorted(target_dict.items(), key=lambda x:x[1], reverse=True)

def read_json(file_dir):
    with open(file_dir, 'rb') as f:
        return json.load(f)

def write_json(file_content, file_dir):
    with open(file_dir, 'w') as f:
        json.dump(file_content, f)
        
def list_avg(target_list):
    return sum(target_list)/len(target_list)