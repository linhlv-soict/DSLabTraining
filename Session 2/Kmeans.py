# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 23:18:34 2020

@author: Linh LV
"""
import numpy as np
from collections import defaultdict

class Member:
    def __init__(self, r_d, label = None, doc_id = None):
        self._r_d = r_d              # bieu dien TF IDF cua van ban d
        self._label = label          # newsgroup cua van ban d
        self._doc_id = doc_id        # ten file chua van ban d
     
class Cluster:
    def __init__(self):
        self._centroid = None
        self._members = []
        
    def reset_members(self):
        self._members = []
        
    def add_members(self, member):
        self._members.append(member)
        
class Kmeans:
    def __init__(self, num_clusters):
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for _ in range(self._num_clusters)]
        self._E = []  # list of centroids
        self._S = 0.  # overall similarity (loi phan cum)
        
    # doc du lieu vao
    def load_data(self, data_path): 
        # chuyen cac gia tri tfidf thanh vector tuong ung
        def sparse_to_dense(sparse_rd, vocab_size):
            rd = [0.0 for _ in range(vocab_size)]
            indices_tfidfs = sparse_rd.split()
            
            for idx in indices_tfidfs:
                index = int(idx.split(':')[0])
                tfidf = float(idx.split(':')[1])
                rd[index] = tfidf
            return np.array(rd)
            
        with open(data_path) as f:
            dlines = f.read().splitlines()
        with open('../datasets/20news-bydate/words_idfs.txt') as f:
            vocab_size = len(f.read().splitlines())
        
        self._data = []
        self._label_count = defaultdict(int)
        for data_id, d in enumerate(dlines):
            features = d.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            self._label_count[label] += 1       # dem so luong member cua moi nhan
            
            rd = sparse_to_dense(sparse_rd = features[2], vocab_size = vocab_size)
            self._data.append(Member(r_d = rd, label = label, doc_id = doc_id))
            
            print(label, ':', doc_id)
            
    #khoi tao ngau nhien cac tam cum
    def random_init(self, seed_val):
        assert seed_val > 0
        if seed_val != self._num_clusters:
            self.__init__(seed_val)
            
        idx = np.random.choice(len(self._data), size = self._num_clusters, replace = False)
        print(idx)
        for i in idx:
            self._E.append(self._data[i]._r_d)
        for index, cluster in enumerate(self._clusters):
            cluster._centroid = (self._E)[index]

    def kmeanspp_init(self, seed_val):
        assert seed_val > 0
        if seed_val != self._num_clusters:
            self.__init__(seed_val)
            
        self._E.append(self._data[np.random.randint(len(self._data))]._r_d)
        while len(self._E) < seed_val:
            x = None
            mins = 1.0
            for member in self._data:
                if not np.array([(member._r_d == centroid).all() for centroid in self._E]).any():
                    pass
                    s = max([self.compute_similarity(member, centroid) for centroid in self._E])
                    if s < mins:
                        mins = s
                        x = member
            self._E.append(x._r_d)
            
        for index, cluster in enumerate(self._clusters):
            cluster._centroid = (self._E)[index]        
        
        
    def compute_similarity(self, member, centroid):
        return np.dot(member._r_d, centroid)  # cosine distance; _rd, centroid are normalized
        
    # chon cluster cho diem du lieu
    def select_cluster(self, member):
        best_fit = None
        max_similarity = -1
        for cluster in self._clusters:
            sm = self.compute_similarity(member, cluster._centroid)
            if sm > max_similarity:
                best_fit = cluster
                max_similarity = sm
        best_fit.add_members(member)  # them phan tu vao cluster moi
        return max_similarity
      
    # cap nhat lai centroid cua cluster
    def update_centroid(self, cluster):
        member_rds = [member._r_d for member in cluster._members]
        aver_rd = np.mean(member_rds, axis = 0)
        sqrt_sum_sqr = np.sqrt(np.sum(aver_rd ** 2))
        new_centroid = np.array([val/sqrt_sum_sqr for val in aver_rd])  # normalized centroid
        
        cluster._centroid = new_centroid
        
    # kiem tra dieu kien dung vong lap    
    def stopping_condition(self, criterion, threshold):
        criteria = ['centroid', 'similarity', 'max_iters']
        assert criterion in criteria
        
        # gioi han so vong lap
        if criterion == 'max_iters':
            if self._iteration >= threshold:
                return True
            else:
                return False
            
        # kiem tra su thay doi cac centroid
        elif criterion == 'centroid':
            E_new = [list(cluster._centroid) for cluster in self._clusters]
            Enew_minus_E = [centroid for centroid in E_new if centroid not in self._E]
            self._E = E_new
            if len(Enew_minus_E) <= threshold:
                return True
            else:
                return False
            
        # kiem tra theo tinh toan loi phan cum
        else:
            newS_minus_S = self._newS - self._S
            self._S = self._newS
            if newS_minus_S <= threshold:
                return True
            else:
                return False
            self._newS = 0.
            for member in self._data:
                maxs = self.select_cluster(member)
                self._newS += maxs
    
    # chay thuat toan phan cum    
    def run(self, seed_val, init, criterion, threshold):
        assert init in ['kmeans++', 'random']
        if init == 'random':
            self.random_init(seed_val)
        elif init == 'kmeans++':
            self.kmeanspp_init(seed_val)
            
        self._iteration = 0
        while True:
            
            print('Iteration: ', self._iteration)
            
            # gan cluster moi cho tat ca member va tinh lai loi phan cum S
            for cluster in self._clusters:
                cluster.reset_members()
            self._newS = 0
            for member in self._data:
                max_s = self.select_cluster(member)
                self._newS += max_s
            print('S = ', self._newS)
            
            # cap nhat lai tam cum moi
            for cluster in self._clusters:
                self.update_centroid(cluster)
             
            # kiem tra dieu kien dung
            self._iteration += 1   
            if self.stopping_condition(criterion, threshold):
                break            
        
    def compute_purity(self):
        major_sum = 0.
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            max_count = max([member_labels.count(label) for label in range(20)])
            major_sum += max_count
        return major_sum * 1./len(self._data)
        
    def compute_NMI(self):
        I_val, H_omg, H_C, N = 0., 0., 0., len(self._data)
        for cluster in self._clusters:
            wk = len(cluster._members) * 1.
            H_omg += -wk / N * np.log10(wk / N)
            member_labels = [member._label for member in cluster._members]
            for label in range(20):
                wk_cj = member_labels.count(label) * 1.
                cj = self._label_count[label]
                I_val += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)
            for label in range(20):
                cj = self._label_count[label] * 1.
                H_C += -cj / N * np.log10(cj / N)
            return I_val * 2. / (H_omg + H_C)
        
if __name__ == '__main__':
    app = Kmeans(num_clusters = 20)
    app.load_data(data_path = '../datasets/20news-bydate/data_tf_idf.txt')
    app.run(seed_val = 20, init = 'random',criterion = 'max_iters', threshold = 50)
    #app.run(seed_val = 20, init = 'kmeans++', criterion = 'max_iters', threshold = 50)
    print('Purity = ', app.compute_purity())
    print('NMI = ', app.compute_NMI())
    
#    S =  2287.011072715092, converged
#    Purity =  0.5194449354781686
#    NMI =  0.11471605835541569