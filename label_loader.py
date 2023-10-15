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

def load_label(label_path):
    char2index = dict() # [ch] = id
    index2char = dict() # [id] = ch
    with open(label_path, 'r') as f:
        for no, line in enumerate(f):
            
            if line[0] == '#': 
                continue

            index, char, freq = line.strip().split('\t')#ID,'값',value
            char = char.strip()
            if len(char) == 0:
                char = ' '

            char2index[char] = int(index) #char2index['군']=0
            index2char[int(index)] = char #index2char['0']=군

    return char2index, index2char
