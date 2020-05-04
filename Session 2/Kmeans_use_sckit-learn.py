# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:04:28 2020

@author: Linh LV
"""
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
        label.append(features[0])
        data.append(sparse_to_dense(features[2], vocab_size))
        
    return data, label

def clustering_with_Kmeans():
    data, label = load_data(data_path = '../datasets/20news-bydate/data_tf_idf.txt')
    
    from sklearn.cluster import KMeans
    from scipy.sparse import csr_matrix
    
    X = csr_matrix(data)
    
    kmeans = KMeans(n_clusters = 20, init = 'k-means++', n_init = 5, tol = 1e-3, random_state = 2000).fit(X)
    labels = kmeans.labels_
    
if __name__ == '__main__':    
    clustering_with_Kmeans()