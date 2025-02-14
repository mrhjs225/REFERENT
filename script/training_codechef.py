from time import time

import os

def get_checkpoint_num(model_dir):
    folder_list = os.listdir(model_dir)
    maximum_num = 0
    for folder in folder_list:
        if folder.startswith('checkpoint-'):
            folder_num = int(folder.split('-')[1])
            if folder_num > maximum_num:
                maximum_num = folder_num
    return str(maximum_num)

epoch = 30
batch = 32
model_name = 't5-base'
mode = '1-1'
pretrained_mode_dir = '../models/t5base/'
train_dataset_dir = '../data/codechef/train_model_{}.json'.format(mode)
save_model_dir = '../models/codechef/codechef_{}'.format(mode)

os.system('rm -rf {}'.format(save_model_dir))
os.system('python tfix_training.py -e {} -bs {} -mn {} -md {} -lm {} -dd {} -bid {} -pt {}'.format(epoch, batch, model_name, save_model_dir, pretrained_mode_dir, train_dataset_dir, 0, 'False'))