# -*- coding: utf-8 -*-
from os import listdir
from os.path import isfile
from collections import defaultdict
import re
# Xay dung tu dien va thu thap du lieu tu files
def gen_data_and_vocab():
    def collect_data_from(parent_path, newsgroup_list, word_count= None):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            dir_path = parent_path + '/' + newsgroup + '/'
            files = [(filename, dir_path + filename)
                     for filename in listdir(dir_path)
                     if isfile(dir_path + filename)]
            files.sort()
            label = group_id
            print('Processing: {}-{}'.format(group_id, newsgroup))
            for filename, filepath in files:
                with open(filepath, 'r', encoding = 'utf-8', errors = 'ignore') as f:
                    text= f.read().lower()
                    words = re.split('\W+', text) # tach cac tu, bo dau cau
                    if word_count is not None: 
                        for word in words:
                            word_count[word] += 1
                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + '<fff>' + filename + '<fff>' + content)
        return data
    
    word_count = defaultdict(int)
    
    path = '../datasets/20news-bydate/'
    parts = [path + dir_name + '/' for dir_name in listdir(path)
            if not isfile(path + dir_name)]
    
    train_path, test_path = (parts[0], parts[1]) \
        if 'train' in parts[0] else (parts[1], parts[0])
            
    newsgroup_list = [newsgroup for newsgroup in listdir(train_path)]
    newsgroup_list.sort()

    train_data = collect_data_from(
        parent_path=train_path,
        newsgroup_list=newsgroup_list,
        word_count=word_count
    )
    
    vocab = [word for word, freq in
             zip(word_count.keys(), word_count.values()) if freq > 10]
    vocab.sort()
    with open('../datasets/w2v/vocab-raw.txt', 'w') as f:
        f.write('\n'.join(vocab))

    test_data = collect_data_from(
        parent_path=test_path,
        newsgroup_list=newsgroup_list
    )

    with open('../datasets/w2v/20news-train-raw.txt', 'w') as f:
        f.write('\n'.join(train_data))
    with open('../datasets/w2v/20news-test-raw.txt', 'w') as f:
        f.write('\n'.join(test_data))
    
def encode_data(data_path, vocab_path, MAX_DOC_LENGTH= 500):
    if 'train' in data_path:
        print ('Encoding training set')
    else:
        print ('Encoding test set')
    with open(vocab_path) as f:
        vocab = dict([(word, word_ID + 2)
                      for word_ID, word in enumerate(f.read().splitlines())])
    
    with open(data_path) as f:
        documents = [(line.split('<fff>')[0], line.split('<fff>')[1], line.split('<fff>')[2])
                     for line in f.read().splitlines()]
    encoded_data = []
    padding_ID = 0  # ID cua tu rong (padding words) trong cac van ban
    unknown_ID = 1  # ID cua tu khong xuat hien trong vocab
    
    # lay thong tin tu cac van ban
    for document in documents:
        label, doc_id, text = document
        words = text.split()[:MAX_DOC_LENGTH]
        sentence_length = len(words)
        
        # thay the word bang ID
        encoded_text = []
        for word in words:
            if word in vocab:
                encoded_text.append(str(vocab[word]))
            else:
                encoded_text.append(str(unknown_ID))
        if len(words) < MAX_DOC_LENGTH:
            num_padding = MAX_DOC_LENGTH - sentence_length
            for _ in range(num_padding):
                encoded_text.append(str(padding_ID))
        encoded_data.append(str(label) + '<fff>' + str(doc_id) + '<fff>' + str(sentence_length)
                            + '<fff>' + ' '.join(encoded_text))
    dir_name = '/'.join(data_path.split('/')[:-1])
    file_name = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + '-encoded.txt'
    with open(dir_name + '/' + file_name, 'w') as f:
        f.write('\n'.join(encoded_data))
    

if __name__ == '__main__':
    #gen_data_and_vocab()
    train_path = '../datasets/w2v/20news-train-raw.txt'
    test_path = '../datasets/w2v/20news-test-raw.txt'
    vocab_path = '../datasets/w2v/vocab-raw.txt'
    encode_data(train_path, vocab_path)
    encode_data(test_path, vocab_path)
    

