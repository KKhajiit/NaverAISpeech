"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

import os
import sys
import math
import wavio
import time
import torch
import random
import threading
import logging
import librosa
import random
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)
rand=0
#np.random.seed(time.time())
PAD = 0
N_FFT = 512
SAMPLE_RATE = 16000

target_dict = dict()

def load_targets(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict[key] = target #target_dict['wav001.wav']=[192 755 662 192 678 476 662 408 690 2 125 610 662 220 640 125 662 179 192 661 123 662]


def manipulateNoise(d,noise_factor):
    noise = np.random.randn(len(d))
    augmented_data = d + noise_factor * noise
    augmented_data = augmented_data.astype(type(d[0]))
    return augmented_data
def manipulatePitch(data,sample_rate,pitch_factor):
    return librosa.effects.pitch_shift(data,sample_rate,pitch_factor)
def manipulateSpeed(data,speed_factor):
    return librosa.effects.time_stretch(data,speed_factor)

def get_spectrogram_feature(filepath):
    global rand 
    rand+=1
    #1 librosa를 이용한 공백 제거
    sig,sr=librosa.core.load(filepath,SAMPLE_RATE)
    sigt,index=librosa.effects.trim(sig,hop_length=128)
    sig=sigt

    """
    if rand%13==0:
        if rand<100:
            print("NOISE MANIPULATE")
        sig=manipulateNoise(sig,0.5)
    if rand%17==0:
        if rand<100:
            print("PITCH MANIPULATE")
        if rand%3==0:
            sig=manipulatePitch(sig,SAMPLE_RATE,2)
        elif rand%3==1:
            sig=manipulatePitch(sig,SAMPLE_RATE,3)
        else:
            sig=manipulatePitch(sig,SAMPLE_RATE,4)
       
    if rand%19==0:
        if rand<100:
            print("SPEED MANIPULATE")
        if rand%9==0:
            sig=manipulateSpeed(sig,0.5)
        elif rand%9==1:
            sig=manipulateSpeed(sig,0.6)
        elif rand%9==2:
            sig=manipulateSpeed(sig,0.7)
        elif rand%9==3:
            sig=manipulateSpeed(sig,0.8)
        elif rand%9==4:
            sig=manipulateSpeed(sig,0.9)
        elif rand%9==5:
            sig=manipulateSpeed(sig,1.1)
        elif rand%9==6:
            sig=manipulateSpeed(sig,1.3)
        elif rand%9==7:
            sig=manipulateSpeed(sig,1.4)
        else:
            sig=manipulateSpeed(sig,1.5)
    """
    #2 특징 추출
    mfcc=librosa.feature.mfcc(y=sig,sr=SAMPLE_RATE,hop_length=128,n_mfcc=40,n_fft=N_FFT)
    #3 정규화
    mean=np.mean(mfcc,axis=0)
    std=np.std(mfcc,axis=0)
    cmvns= ((mfcc-mean)+1e-8)/std
    cmvns=torch.FloatTensor(cmvns).transpose(0,1)
    mfcc=torch.FloatTensor(mfcc).transpose(0,1)
    allfeature=torch.cat([mfcc,cmvns],dim=1)
    return allfeature

def get_real_feature(filepath):
    sig,sr=librosa.core.load(filepath,SAMPLE_RATE)
    #1 공백 제거
    sigt,index=librosa.effects.trim(sig,hop_length=128)
    sig=sigt
    #특징추출
    mfcc=librosa.feature.mfcc(y=sig,sr=SAMPLE_RATE,hop_length=128,n_mfcc=40,n_fft=N_FFT)
    #표준화
    mean=np.mean(mfcc,axis=0)
    std=np.std(mfcc,axis=0)
    cmvns= ((mfcc-mean)+1e-8)/std
    cmvns=torch.FloatTensor(cmvns).transpose(0,1)
    mfcc=torch.FloatTensor(mfcc).transpose(0,1)
    allfeature=torch.cat([mfcc,cmvns],dim=1)
    return allfeature
    
def get_script(filepath, bos_id, eos_id):
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')
    result = list()
    result.append(bos_id)
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            result.append(int(tokens[i]))
    result.append(eos_id)
    return result

class BaseDataset(Dataset):
    def __init__(self, wav_paths, script_paths, bos_id=1307, eos_id=1308):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.bos_id, self.eos_id = bos_id, eos_id

    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        feat = get_spectrogram_feature(self.wav_paths[idx])
        script = get_script(self.script_paths[idx], self.bos_id, self.eos_id)
        return feat,script
    

def _collate_fn(batch):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(PAD)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    return seqs, targets, seq_lengths, target_lengths

class BaseDataLoader(threading.Thread):
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)
        seq_lengths = list()
        target_lengths = list()
        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        logger.debug('loader %d start' % (self.thread_id))
        while True:
            items = list()

            for i in range(self.batch_size): 
                if self.index >= self.dataset_count:
                    break

                items.append(self.dataset.getitem(self.index))
                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)
            self.queue.put(batch)
        logger.debug('loader %d stop' % (self.thread_id))

class MultiLoader():
    def __init__(self, dataset_list, queue, batch_size, worker_size):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_size = worker_size
        self.loader = list()

        for i in range(self.worker_size):
            self.loader.append(BaseDataLoader(self.dataset_list[i], self.queue, self.batch_size, i))

    def start(self):
        for i in range(self.worker_size):
            self.loader[i].start()

    def join(self):
        for i in range(self.worker_size):
            self.loader[i].join()

