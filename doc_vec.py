# -*- coding: utf-8 -*-
"""
Created on Sat May 8 2018.
@author: Liu Aiting
@e-mail: liuaiting@bupt.edu.cn
"""
import codecs
import time
import collections
from multiprocessing import Pool

import csv
import jieba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config


def read_stop_words(path=config.PATH_STOP_WORDS):
    # remove common words and tokenize
    stopwords = []
    f = codecs.open(path, 'r', 'utf-8', errors='ignore')
    for word in f:
        w = word.strip()
        if w:
            stopwords.append(w)
    stop_list = set(stopwords)
    return stop_list


def jieba_cut(path_corpus_raw, path_corpus_cut, path_corpus, del_stop_words=False):
    # Use jieba to cut raw corpus into word sequence.
    with codecs.open(path_corpus_raw, 'r', 'utf-8', errors='ignore') as f:
        with codecs.open(path_corpus_cut, 'w', 'utf-8') as fw:
            with codecs.open(path_corpus, 'w', 'utf-8') as ft:
                reader = csv.reader(f)
                for row in reader:
                    line = row[1].strip().lower()
                    word_list = jieba.cut(line, cut_all=False)
                    if del_stop_words:
                        stop_list = read_stop_words()
                        word_list = [word for word in word_list if word not in stop_list]
                    sent = ' '.join(word_list)
                    ft.write(sent + '\n')
                    new_row = [row[0], sent]
                    fw.write('\t'.join(new_row) + '\n')


# load word2vec
def load_word2vec(path=config.PATH_WORD2VEC):
    start_time = time.time()
    print("Loading word vectors from {}.".format(path))
    word2id = collections.OrderedDict()
    id2word = []
    id2vec = []
    with codecs.open(path, 'r', 'utf-8') as f:
        lines = f.readlines()[1:]
        for i, line in enumerate(lines):
            values = line.strip().split()
            word = values[0]
            vector = values[1:]
            id2vec.append([float(v) for v in vector])
            word2id[word] = i
            id2word.append(word)
    print("Loading word2vec cost {}s.".format(time.time()-start_time))
    return word2id, id2word, id2vec


def file_to_id(path_cut, path_cut_id, word2id):
    start_time = time.time()
    print("Transform word sequence to ids.")
    with codecs.open(path_cut, 'r', 'utf-8') as f:
        with codecs.open(path_cut_id, 'w', 'utf-8') as fw:
            for line in f:
                row = line.strip().split('\t')
                title = row[1].strip().split()
                title2id = [str(word2id[w]) for w in title]
                new_row = [row[0], ' '.join(title2id)]
                fw.write('\t'.join(new_row) + '\n')
    print("{} -> {} cost {}s.".format(path_cut, path_cut_id, time.time()-start_time))


def file_to_vec(path_cut_id, path_doc2vec, id2vec):
    start_time = time.time()
    print("Transform ids to vectors and get doc_vecs.")
    with codecs.open(path_cut_id, 'r', 'utf-8') as f:
        with codecs.open(path_doc2vec, 'w', 'utf-8') as fw:
            for line in f:
                row = line.strip().split('\t')
                ids = [int(i) for i in row[1].strip().split()]
                word_vec = [id2vec[i] for i in ids]
                doc_vec = np.average(word_vec, axis=0)
                res = [row[0], ' '.join([str(i) for i in doc_vec])]
                fw.write('\t'.join(res) + '\n')
    print("{} -> {} cost {}s.".format(path_cut_id, path_doc2vec, time.time()-start_time))


def get_cosine(s1, s2):
    s = cosine_similarity(s1, s2)
    # pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2))
    # return s1[pos_a:pos_a+size]
    return s[0][0]


def callback(results):
    with codecs.open(config.PATH_RESULT_DOC2VEC, 'a+', 'utf-8') as f:
        for res in results:
            res = [str(r) for r in res]
            f.write('\t'.join(res) + '\n')

    with codecs.open(config.PATH_RESULT_DOC2VEC_TOP20, 'a+', 'utf-8') as f20:
        top20 = sorted(results, key=lambda x: x[2], reverse=True)[:21]
        num = 0
        for res in top20:
            if res[0] != res[1] and num <= 20:
                res = [str(r) for r in res]
                f20.write('\t'.join(res) + '\n')
                num += 1


def get_cosine_file(row_test):
    print('Processing {}.'.format(row_test[0]))
    results = []
    with codecs.open(config.PATH_CORPUS_TRAIN_DOC2VEC, 'r', 'utf-8') as f:
        for line in f:
            row_train = line.strip().split('\t')
            source_id = int(row_test[0])
            target_id = int(row_train[0])
            source_vec = [[float(v) for v in row_test[1].strip().split()]]
            target_vec = [[float(v) for v in row_train[1].strip().split()]]
            similarity = float(get_cosine(source_vec, target_vec))
            res = [source_id, target_id, similarity]
            results.append(res)
    return results


if __name__ == '__main__':
    start_time = time.time()
    print("==================================")
    jieba_cut(config.PATH_CORPUS_TRAIN_RAW, config.PATH_CORPUS_TRAIN_CUT, config.PATH_CORPUS_TRAIN)
    jieba_cut(config.PATH_CORPUS_TEST_RAW, config.PATH_CORPUS_TEST_CUT, config.PATH_CORPUS_TEST)
    tmp_time = time.time()
    print("Jieba cut cost {}s.".format(tmp_time-start_time))
    print("==================================")
    word_to_id, id_to_word, id_to_vec = load_word2vec()

    file_to_id(config.PATH_CORPUS_TRAIN_CUT, config.PATH_CORPUS_TRAIN_CUT_ID, word_to_id)
    file_to_id(config.PATH_CORPUS_TEST_CUT, config.PATH_CORPUS_TEST_CUT_ID, word_to_id)
    #
    file_to_vec(config.PATH_CORPUS_TRAIN_CUT_ID, config.PATH_CORPUS_TRAIN_DOC2VEC, id_to_vec)
    file_to_vec(config.PATH_CORPUS_TEST_CUT_ID, config.PATH_CORPUS_TEST_DOC2VEC, id_to_vec)
    print("Preprocessing cost {}s.".format(time.time()-tmp_time))
    print("==================================")

    tmp_time = time.time()
    rows = []
    with codecs.open(config.PATH_CORPUS_TEST_DOC2VEC, 'r', 'utf-8') as ftest:
        for line in ftest:
            row = line.strip().split('\t')
            rows.append(row)
    p = Pool(70)
    for row in rows:
        p.apply_async(get_cosine_file, (row,), callback=callback)
    p.close()
    p.join()
    print('Get cosine similarity cost %.4fs.' % (time.time()-tmp_time))
    print("==================================")

    tmp_time = time.time()
    with codecs.open(config.PATH_RESULT_DOC2VEC_TOP20, 'r', 'utf-8') as f:
        with codecs.open(config.PATH_RESULT_DOC2VEC_TOP20_SORTED, 'w', 'utf-8') as fw:
            fw.write('source_id\ttarget_id\tsimilarity\n')
            rows = []
            for row in f:
                row = row.strip().split('\t')
                source_id = int(row[0])
                target_id = int(row[1])
                similarity = float(row[2])
                rows.append([source_id, target_id, similarity])
            results = sorted(rows, key=lambda x: x[2], reverse=True)
            results = sorted(results, key=lambda x: x[0])
            for res in results:
                fw.write('\t'.join([str(r) for r in res]) + '\n')

    print("Get cosine similarity top20 result cost {}s.".format(time.time()-tmp_time))


