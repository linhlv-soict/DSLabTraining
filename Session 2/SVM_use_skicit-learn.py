# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:36:54 2020

@author: Linh LV
"""
import numpy as np
def sparse_to_dense(sparse_rd, vocab_size):
    rd = [0.0 for _ in range(vocab_size)]
    indices_tfidfs = sparse_rd.split()
            
    for idx in indices_tfidfs:
        index = int(idx.split(':')[0])
        tfidf = float(idx.split(':')[1])
        rd[index] = tfidf
    return rd
            

def load_data(data_path):
    with open('../datasets/20news-bydate/words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())
    with open(data_path) as f:
        dlines = f.read().splitlines()
    label = []
    data = []
    
    for line in dlines:
        features = line.split('<fff>')
        label.append(int(features[0]))
        data.append(sparse_to_dense(features[2], vocab_size))
        
    return np.array(data), np.array(label)

def compute_accuracy(predicted_y, expected_y):
    matches = np.equal(predicted_y, expected_y)
    acc = np.sum(matches.astype(float)) / expected_y.size
    return acc

def classifying_with_linear_SVMs(train_X, train_Y, test_X, test_y):
    #train_X, train_Y = load_data(data_path = '../datasets/20news-bydate/20news-train-tfidf.txt')
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(
            C = 10.0,       # penalty coefficient
            tol = 0.001,    # tolerance for stopping criteria
            verbose = True  # whether prints out logs or not
            )
    classifier.fit(train_X, train_Y)
    
    #test_X, test_y = load_data(data_path = '../datasets/20news-bydate/20news-test-tfidf.txt')
    predicted_y = classifier.predict(test_X)
    
    accuracy = compute_accuracy(predicted_y, test_y)
    print()
    print ('Accuracy Linear SVMs: ', accuracy)

def classifying_with_kernel_SVMs(train_X, train_Y, test_X, test_y, _kernel = 'rbf'):
    from sklearn.svm import SVC
    classifier = SVC(
            C = 50.0,         # penalty coefficient
            kernel = _kernel, #kernel function
            gamma = 'scale',
            tol = 0.001,      # tolerance for stopping criteria
            verbose = True,    # whether prints out logs or not
            max_iter = 100
            )
    classifier.fit(train_X, train_Y)
    predicted_y = classifier.predict(test_X)
    
    accuracy = compute_accuracy(predicted_y, test_y)
    print()
    print ('Accuracy Kernel SVMs: ', accuracy)
    
if __name__ == '__main__':
    train_X, train_Y = load_data(data_path = '../datasets/20news-bydate/20news-train-tfidf.txt')
    test_X, test_y = load_data(data_path = '../datasets/20news-bydate/20news-test-tfidf.txt')
    classifying_with_linear_SVMs(train_X, train_Y, test_X, test_y)
    classifying_with_kernel_SVMs(train_X, train_Y, test_X, test_y, _kernel = 'linear')
    
    # Ket qua:
    # Linear SVMs: accuracy = 0.8255443441317047
    # Kernel SVMs (radial basis function kernel): accuracy = 