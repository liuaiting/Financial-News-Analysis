# -*- coding:utf-8 -*-
"""
Created on Thurs Sep 21 2017.
@author: Liu Aiting
@e-mail: liuaiting@bupt.edu.cn

"""
from __future__ import print_function
from __future__ import division

import codecs
import csv
import time
from multiprocessing import Pool

from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint  # pretty-printer

import config

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_topics = config.num_topics


def preprocessing(documents, path_stop_words=config.PATH_STOP_WORDS):
    # remove common words and tokenize
    stopwords = []
    f = codecs.open(path_stop_words, 'r', 'utf-8', errors='ignore')
    for word in f:
        w = word.strip()
        if w:
            stopwords.append(w)
    stoplist = set(stopwords)
    # print(stoplist)
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]
    pprint(len(texts))
    return texts


def strings_to_vectors(path_corpus):
    start_time = time.time()
    f = codecs.open(path_corpus, 'r', 'utf-8', errors='ignore')
    documents = []
    for line in f:
        documents.append(line.strip().split())
    pprint(len(documents))

    dictionary = corpora.Dictionary(documents)
    dictionary.save(config.PATH_WORDS_DIC)  # store the dictionary, for future reference

    corpus = [dictionary.doc2bow(text) for text in documents]
    corpora.MmCorpus.serialize(config.PATH_MM_CORPUS, corpus)  # store to disk, for later use

    end_time = time.time()
    print("String to vectors cost %.4f s." % (end_time-start_time))


def transformation():
    start_time = time.time()

    # Creating a transformation
    tfidf = models.TfidfModel(corpus)  # initialize a model

    # Transforming vectors
    corpus_tfidf = tfidf[corpus]

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=config.num_topics)  # initialize an LSI transformation
    corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    lsi.print_topics(20)
    # for doc in corpus_lsi:  # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    #     print(doc)
    lsi.save(config.PATH_LSI_MODEL)  # same for tfidf, lda, ...
    end_time = time.time()
    print("Transformation cost %.4f s." % (end_time-start_time))


def similarity():

    start_time = time.time()
    # dictionary = corpora.Dictionary.load(PATH_WORDS_DIC)
    # corpus = corpora.MmCorpus.load(PATH_MM_CORPUS)  # comes from the first tutorial, "From strings to vectors"
    # lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=20)
    lsi = models.LsiModel.load(config.PATH_LSI_MODEL)
    similar_model = similarities.MatrixSimilarity(lsi[corpus], num_best=None)
    similar_model.save(config.PATH_LSI_INDEX_MODEL)
    end_time = time.time()
    print("Similarity cost %.4f s." % (end_time-start_time))


def get_lsi_sim(row_test):
    results = []
    source_id = int(row_test[0])
    doc = row_test[1].strip()
    print(doc)
    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lsi = lsi[vec_bow]
    sims = similar_model[vec_lsi]  # perform a similarity query against the corpus
    # sims = sorted(sims, key=lambda x: x[0])
    # print(sims)
    for i, sim in enumerate(sims):
        target_id = int(i + 1)
        similarity_value = float(sim)
        res = [source_id, target_id, similarity_value]
        results.append(res)
    # print(source_id, results)
    print(len(results))
    return results


def mycallback(results):
    with codecs.open(config.PATH_RESULT_LSI, 'a+', 'utf-8') as fa:
        # writer = tsv.TsvWriter(f)
        for res in results:
            res = [str(r) for r in res]
            fa.write('\t'.join(res) + '\n')

    with codecs.open(config.PATH_RESULT_LSI_TOP20, 'a+', 'utf-8') as f20:
        # writer = tsv.TsvWriter(f20)
        top20 = sorted(results, key=lambda x: x[2], reverse=True)[:21]
        num = 0
        for res in top20:
            if int(res[0]) != int(res[1]) and num <= 20:
                res = [str(r) for r in res]
                f20.write('\t'.join(res) + '\n')
                num += 1


def postprocessing(path_result, path_result_top20):
    with codecs.open(path_result, 'r', 'utf-8') as f:
        with codecs.open(path_result_top20, 'w', 'utf-8') as fw:
            fw.write('source_id\ttarget_id\tsimilarity\n')
            reader = csv.reader(f)
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
            for result, row in zip(results, reader):
                top20 = 0
                for res in result:
                    if str(row[0]) != str(res[0]+1) and top20 <= 20:
                        # print(row[0], res[0]+1)
                        fw.write(str(row[0] + '\t' + str(res[0]+1) + '\n'))
                        top20 += 1


def interface():
    # lsi = models.LsiModel.load(config.PATH_LSI_MODEL)
    # similar_model = similarities.MatrixSimilarity.load(config.PATH_LSI_INDEX_MODEL)
    # _input = input("Please input a relevant query：（you can reference 'corpus_train.txt'）")
    # doc = ' '.join(jieba.cut(_input))
    doc = "日本 1 月 失业率 降至 1993 年 4 月 以来 最低水平 。"
    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lsi = lsi[vec_bow]
    # print(vec_lsi)
    sims = similar_model[vec_lsi]  # perform a similarity query against the corpus
    # print(list(enumerate(sims)))  # print (document_number, document_similarity) 2-tuples
    # sims = sorted(sims, key=lambda x: x[0])
    print(len(sims))
    print(sims[:100])


if __name__ == '__main__':

    # jieba_cut(PATH_CORPUS_TRAIN_RAW, PATH_CORPUS_TRAIN)
    # jieba_cut(PATH_CORPUS_TEST_RAW, PATH_CORPUS_TEST)
    # strings_to_vectors(config.PATH_CORPUS_TRAIN, config.PATH_STOP_WORDS)

    dictionary = corpora.Dictionary.load(config.PATH_WORDS_DIC)
    corpus = corpora.MmCorpus(config.PATH_MM_CORPUS)
    transformation()
    similarity()
    lsi = models.LsiModel.load(config.PATH_LSI_MODEL)
    similar_model = similarities.MatrixSimilarity.load(config.PATH_LSI_INDEX_MODEL)

    tmp_time = time.time()
    rows = []
    with codecs.open(config.PATH_CORPUS_TEST_CUT, 'r', 'utf-8',) as ftest:
        # test_reader = csv.reader(ftest)
        for line in ftest:
            row = line.strip().split('\t')
            rows.append(row)
    p = Pool(70)
    for row in rows:
        p.apply_async(get_lsi_sim, (row, ), callback=mycallback)
    p.close()
    p.join()
    print('Get cosine similarity cost %.4fs.' % (time.time()-tmp_time))

    tmp_time = time.time()
    with codecs.open(config.PATH_RESULT_LSI_TOP20, 'r', 'utf-8') as f:
        with codecs.open(config.PATH_RESULT_LSI_TOP20_SORTED, 'w', 'utf-8') as fw:
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



