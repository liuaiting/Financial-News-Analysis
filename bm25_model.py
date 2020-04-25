# -*- coding:utf-8 -*-
"""
Created on Wed May 9 2018.
@author: Liu Aiting
@e-mail: liuaiting@bupt.edu.cn
"""
from __future__ import print_function
from __future__ import division

import codecs
import pickle
import time
import csv
import re
from multiprocessing import Pool

import jieba
import thulac
import pynlpir
from gensim import corpora
from gensim.summarization import bm25

import config

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_stop_words(path=config.PATH_STOP_WORDS):
    """

    :param path:
    :return:
    """
    # remove common words and tokenize
    print("Loading stop words...")
    stopwords = codecs.open(path, 'r', 'utf-8', errors='ignore').readlines()
    stopwords = [w.strip() for w in stopwords]
    return stopwords


def char_tokenizer(sentence):
    """Cut the sentence into the format we want:
    - continuous letters and symbols like back-slash and parenthese
    - single Chinese character
    - other symbols
    """
    regex = []
    # English and number part for type name.
    regex += [r'[0-9a-zA-Z\\+()\-<>]+']
    # Chinese characters part.
    regex += [r'[\u4e00-\ufaff]']
    # Exclude the space.
    regex += [r'[^\s]']
    regex = '|'.join(regex)
    _RE = re.compile(regex)
    segs = _RE.findall(sentence.strip())
    return segs


def thu_tokenizer(sentence):
    thu1 = thulac.thulac(seg_only=True)
    text = thu1.cut(sentence, text=True)
    segs = text.split()
    return segs


def nlpir_tokenizer(sentence):
    pynlpir.open()
    segs = pynlpir.segment(sentence, pos_tagging=False)
    pynlpir.close()
    return segs


def rewrite(line, tokenizer=None):
    line = line.strip().lower()
    line = tokenizer(line)
    line = ' '.join(line)
    return line


def jieba_cut(path_corpus_raw, path_corpus_bm25, stopwords=None):
    # Use jieba to cut raw corpus into word sequence.
    print("Use jieba to cut raw corpus {}.".format(path_corpus_raw))
    with codecs.open(path_corpus_raw, 'r', 'utf-8', errors='ignore') as f:
        with codecs.open(path_corpus_bm25, 'w', 'utf-8') as fw:
            reader = csv.reader(f)
            for row in reader:
                line = row[1].strip().lower()
                # tmp = line.split('：')
                # if len(tmp) > 1:
                #     line = ''.join(tmp[1:])
                # else:
                #     line = line
                word_list = jieba.cut(line, cut_all=False)
                # word_list = jieba.cut_for_search(line)
                # word_list = [word for word in word_list if word not in stopwords]
                sent = ' '.join(word_list)
                new_row = [row[0], sent]
                fw.write('\t'.join(new_row) + '\n')
    print("Saved in {}.".format(path_corpus_bm25))


def char_cut(path_corpus_raw, path_corpus_bm25, stopwords=None):
    # Use self defined tokenizer to cut raw corpus into word sequence.
    print("Use self defined tokenizer to cut raw corpus {}.".format(path_corpus_raw))
    with codecs.open(path_corpus_raw, 'r', 'utf-8', errors='ignore') as f:
        with codecs.open(path_corpus_bm25, 'w', 'utf-8') as fw:
            reader = csv.reader(f)
            for row in reader:
                sent = row[1].strip()
                sent = rewrite(sent, tokenizer=char_tokenizer)
                # word_list = [word for word in word_list if word not in stopwords]
                # sent = ' '.join(word_list)
                new_row = [row[0], sent]
                fw.write('\t'.join(new_row) + '\n')
    print("Saved in {}.".format(path_corpus_bm25))


def thu_cut(path_corpus_raw, path_corpus_bm25, stopwords=None):
    # Use thulac to cut raw corpus into word sequence.
    thu1 = thulac.thulac(seg_only=True)
    print("Use thulac to cur raw corpus {}.".format(path_corpus_raw))
    with codecs.open(path_corpus_raw, 'r', 'utf-8', errors='ignore') as f:
        with codecs.open(path_corpus_bm25, 'w', 'utf-8') as fw:
            reader = csv.reader(f)
            for row in reader:
                sent = row[1].strip()
                sent = thu1.cut(sent, text=True)
                # word_list = [word for word in word_list if word not in stopwords]
                # sent = ' '.join(word_list)
                new_row = [row[0], sent]
                fw.write('\t'.join(new_row) + '\n')
    print("Saved in {}.".format(path_corpus_bm25))


def nlpir_cut(path_corpus_raw, path_corpus_bm25, stopwords=None):
    # Use NLPIR to cut raw corpus into word sequence.
    pynlpir.open()
    print("Use NLPIR to cur raw corpus {}.".format(path_corpus_raw))
    with codecs.open(path_corpus_raw, 'r', 'utf-8', errors='ignore') as f:
        with codecs.open(path_corpus_bm25, 'w', 'utf-8') as fw:
            reader = csv.reader(f)
            for row in reader:
                sent = row[1].strip()
                word_list = pynlpir.segment(sent, pos_tagging=False)
                # word_list = [word for word in word_list if word not in stopwords]
                sent = ' '.join(word_list)
                new_row = [row[0], sent]
                fw.write('\t'.join(new_row) + '\n')
    print("Saved in {}.".format(path_corpus_bm25))
    pynlpir.close()


def creat_corpus(path_train_bm25, path_corpus, path_dictionary):
    print("Creating corpus and dictionary...")
    corpus = []
    i = 0
    with codecs.open(path_train_bm25, 'r', 'utf-8') as f:
        for line in f:
            if line.strip():
                row = line.strip().split('\t')
                try:
                    corpus.append(row[1].strip().split())
                except IndexError:
                    print(row)
                    corpus.append([''])
                i += 1
            if i % 10000 == 0:
                print(i)
    print(len(corpus))
    dictionary = corpora.Dictionary(corpus)
    pickle.dump(corpus, open(path_corpus, 'wb'))
    pickle.dump(dictionary, open(path_dictionary, 'wb'))
    return corpus, dictionary


def load_corpus(path_corpus, path_dictionary):
    print("Loading corpus and dictionary...")
    corpus = pickle.load(open(path_corpus, 'rb'))
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    return corpus, dictionary


def creat_model(path_model, path_avgidf):
    print("Creating bm25 model...")
    bm25Model = bm25.BM25(corpus)
    average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
    pickle.dump(bm25Model, open(path_model, 'wb'))
    pickle.dump(average_idf, open(path_avgidf, 'wb'))
    return bm25Model, average_idf


def load_model(path_model, path_avgidf):
    print('Loading bm25 model...')
    bm25Model = pickle.load(open(path_model, 'rb'))
    average_idf = pickle.load(open(path_avgidf, 'rb'))
    return bm25Model, average_idf


def get_bm25_sim(row_test):
    results = []
    source_id = int(row_test[0])
    doc = row_test[1].strip().split()
    # print(doc)
    scores = bm25Model.get_scores(doc, average_idf)
    # print(scores[:100])
    for idx, sim in enumerate(scores):
        target_id = int(idx + 1)
        similarity_value = float(sim)
        res = [source_id, target_id, similarity_value]
        results.append(res)
    # print(source_id, results)
    print(source_id, len(results))
    return results


def mycallback(results):
    with codecs.open(config.PATH_RESULT_BM25_JIEBA, 'a+', 'utf-8') as fa:
        # writer = tsv.TsvWriter(f)
        for res in results:
            res = [str(r) for r in res]
            fa.write('\t'.join(res) + '\n')

    with codecs.open(config.PATH_RESULT_BM25_JIEBA_TOP20, 'a+', 'utf-8') as f20:
        # writer = tsv.TsvWriter(f20)
        top20 = sorted(results, key=lambda x: x[2], reverse=True)[:21]
        num = 0
        for res in top20:
            if int(res[0]) != int(res[1]) and num <= 20:
                res = [str(r) for r in res]
                f20.write('\t'.join(res) + '\n')
                num += 1


if __name__ == '__main__':
    start = time.time()
    # stop_list = read_stop_words()
    jieba_cut(config.PATH_CORPUS_TEST_RAW, config.PATH_CORPUS_TEST_BM25_JIEBA, stopwords=None)
    jieba_cut(config.PATH_CORPUS_TRAIN_RAW, config.PATH_CORPUS_TRAIN_BM25_JIEBA, stopwords=None)
    # char_cut(config.PATH_CORPUS_TEST_RAW, config.PATH_CORPUS_TEST_BM25)
    # char_cut(config.PATH_CORPUS_TRAIN_RAW, config.PATH_CORPUS_TRAIN_BM25)
    # thu_cut(config.PATH_CORPUS_TEST_RAW, config.PATH_CORPUS_TEST_BM25_THU)
    # thu_cut(config.PATH_CORPUS_TRAIN_RAW, config.PATH_CORPUS_TRAIN_BM25_THU)
    # nlpir_cut(config.PATH_CORPUS_TEST_RAW, config.PATH_CORPUS_TEST_BM25_NLPIR)
    # nlpir_cut(config.PATH_CORPUS_TRAIN_RAW, config.PATH_CORPUS_TRAIN_BM25_NLPIR)

    corpus, dictionary = creat_corpus(config.PATH_CORPUS_TRAIN_BM25_JIEBA, config.PATH_BM25_JIEBA_CORPUS, config.PATH_BM25_JIEBA_DICTIONARY)
    # corpus, dictionary = load_corpus(config.PATH_BM25_JIEBA_CORPUS, config.PATH_BM25_JIEBA_DICTIONARY)
    bm25Model, average_idf = creat_model(config.PATH_BM25_JIEBA_MODEL, config.PATH_BM25_JIEBA_MODEL_AVGIDF)
    # bm25Model, average_idf = load_model(config.PATH_BM25_JIEBA_MODEL, config.PATH_BM25_JIEBA_MODEL_AVGIDF)
    print(bm25Model)
    print(average_idf)

    tmp_time = time.time()
    rows = []
    with codecs.open(config.PATH_CORPUS_TEST_BM25_JIEBA, 'r', 'utf-8',) as ftest:
        # test_reader = csv.reader(ftest)
        for line in ftest:
            row = line.strip().split('\t')
            rows.append(row)
    p = Pool(50)
    for row in rows:
        p.apply_async(get_bm25_sim, (row, ), callback=mycallback)
    p.close()
    p.join()
    print('Get cosine similarity cost %.4fs.' % (time.time()-tmp_time))

    tmp_time = time.time()
    with codecs.open(config.PATH_RESULT_BM25_JIEBA_TOP20, 'r', 'utf-8') as f:
        with codecs.open(config.PATH_RESULT_BM25_JIEBA_TOP20_SORTED, 'w', 'utf-8') as fw:
            fw.write('source_id\ttarget_id\tsimilarity\n')
            rows = []
            for row in f:
                row = row.strip().split('\t')
                source_id = int(row[0])
                target_id = int(row[1])
                sim = float(row[2])
                rows.append([source_id, target_id, sim])
            results = sorted(rows, key=lambda x: x[2], reverse=True)
            results = sorted(results, key=lambda x: x[0])
            for res in results:
                fw.write('\t'.join([str(r) for r in res]) + '\n')
    print("Get cosine similarity top20 result cost {}s.".format(time.time()-tmp_time))
    print(time.time()-start)


