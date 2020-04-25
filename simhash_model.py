# -*-coding:utf-8-*-
"""
Date: 2016-10-24
Author: Aiting Liu
e-mail: liuaiting@bupt.edu.cn
Location: Center for Intelligence of Science and Technology(CIST)@Beijing University of Posts and Telecommunications
"""
from __future__ import division, unicode_literals
import codecs
import time
import pickle

import jieba
import jieba.analyse
from simhash import Simhash, SimhashIndex
from sklearn.feature_extraction.text import TfidfVectorizer

import config


def get_features(s):
    """Get contents features."""
    result = {}
    # get feature weight pair
    for x, w in jieba.analyse.extract_tags(s, topK=100, withWeight=True):
        result[x] = w
    return result.keys()


def process(data_file_path):
    # data = {}
    data = []
    with codecs.open(data_file_path, 'r', 'utf-8') as f:
        for line in f:
            row = line.strip().split('\t')
            # target_id = int(row[0])
            target_title = row[1]
            # data[target_id] = target_title
            data.append(target_title)

    # vec = TfidfVectorizer()
    # D = vec.fit_transform(data)
    # voc = dict((i, w) for w, i in vec.vocabulary_.items())
    # pickle.dump(D, open(config.PATH_SIMHASH_MODEL_D, 'wb'))
    # pickle.dump(voc, open(config.PATH_SIMHASH_MODEL_VOC, 'wb'))
    print("Loading tfidf model...")
    D = pickle.load(open(config.PATH_SIMHASH_MODEL_D, 'rb'))
    voc = pickle.load(open(config.PATH_SIMHASH_MODEL_VOC, 'rb'))

    shs = []
    count = 0
    for i in range(D.shape[0]):
        Di = D.getrow(i)
        # features as list of (token, weight) tuples)
        features = zip([voc[j] for j in Di.indices], Di.data)
        shs.append(Simhash(features))
        count += 1
        if count % 10000 == 0:
            print('Processing line {}.'.format(count))

    test_ids = open(config.PATH_CORPUS_TEST_ID, 'r').readlines()
    test_ids = [int(i.strip()) for i in test_ids]

    with open(config.PATH_RESULT_SIMHASH_SORTED, 'w') as f1:
        with open(config.PATH_RESULT_SIMHASH_TOP20_SORTED, 'w') as f2:
            f2.write('source_id\ttarget_id\n')
            for source_id in test_ids:
                print("source id : {}".format(source_id))
                results = []
                for index in range(D.shape[0]):
                    source_index = int(source_id - 1)
                    dis = shs[source_index].distance(shs[index])
                    target_id = int(index) + 1
                    res = [source_id, target_id, dis]
                    results.append(res)
                results = sorted(results, key=lambda x: x[2])
                for res in results:
                    f1.write('\t'.join([str(r) for r in res]) + '\n')
                top20 = results[:21]
                num = 0
                for res in top20:
                    if res[0] != res[1] and num <= 20:
                        f2.write('\t'.join([str(r) for r in res[:2]]) + '\n')


if __name__ == '__main__':

    start = time.time()
    process(config.PATH_CORPUS_TRAIN_CUT)

    end = time.time()
    print("Simhash cost : %.03f seconds ......" %(end-start))













