# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 23:21:55 2020

@author: Linh LV
"""
import random
import numpy as np

class DataReader:
    # Read data from file
    def __init__(self, data_path, batch_size):
        
        self._batch_size = batch_size
        
        
        with open(data_path) as f:
            d_lines = f.read().splitlines()
        self._data = []
        self._labels = []
        self._sentence_lengths = []
        
        for data_id, line in enumerate(d_lines):
           
            features = line.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            tokens = features[3].split()

            vector = [int(_) for _ in tokens] 
            
            self._data.append(vector)
            self._labels.append(label)
            self._sentence_lengths.append(int(features[2]))
            

        self._data = np.array(self._data)
        self._labels = np.array(self._labels)
        self._sentence_lengths = np.array(self._sentence_lengths)
        
        self._num_epoch = 0
        self._batch_id = 0
    
    # Divide into batches
    def next_batch(self):
        start = self._batch_id * self._batch_size
        end = start + self._batch_size
        
        self._batch_id += 1
        
        
        if end > len(self._data):
            start = len(self._data) - self._batch_size
            end = len(self._data)
            data, labels, sentence_length = self._data[start:end], self._labels[start:end], self._sentence_lengths[start:end]
            
            self._num_epoch += 1
            self._batch_id = 0
            indices = list(range(len(self._data)))
            random.seed(2020)
            random.shuffle(indices)
            self._data, self._labels = self._data[indices], self._labels[indices]
        else:
            data, labels, sentence_length = self._data[start:end], self._labels[start:end], self._sentence_lengths[start:end]
        
        self._current_part = self._batch_id
            
        return data, labels, sentence_length