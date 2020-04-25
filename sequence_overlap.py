# -*- coding: utf-8 -*-
"""
Created on Sat May 8 2018.
@author: Liu Aiting
@e-mail: liuaiting@bupt.edu.cn
"""
import difflib
import codecs
import time
from multiprocessing import Pool

import csv

import utils
import config


def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    # pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2))
    # return s1[pos_a:pos_a+size]
    return s.ratio()


def callback(results):
    with codecs.open(config.PATH_RESULT_OVERLAP, 'a+', 'utf-8') as f:
        for res in results:
            res = [str(r) for r in res]
            f.write('\t'.join(res) + '\n')

    with codecs.open(config.PATH_RESULT_OVERLAP_TOP20, 'a+', 'utf-8') as f20:
        top20 = sorted(results, key=lambda x: x[2], reverse=True)[0:21]
        num = 0
        for res in top20:
            if res[0] != res[1] and num <= 20:
                res = [str(r) for r in res]
                f20.write('\t'.join(res) + '\n')
                num += 1


def get_overlap_file(row_test):
    print("Processing {}.".format(row_test[0]))
    results = []
    train_reader = csv.reader(open(config.PATH_CORPUS_TRAIN_RAW, 'r'))
    for row_train in train_reader:
        # print(row_train)
        source_id = int(row_test[0])
        target_id = int(row_train[0])
        source_title = utils.rewrite(row_test[1])
        target_title = utils.rewrite(row_train[1])
        similarity = float(get_overlap(source_title, target_title))
        res = [source_id, target_id, similarity]
        results.append(res)
    # for i in range(5):
    #     print(results[i])
    return results


if __name__ == '__main__':
    start_time = time.time()
    rows = []
    with codecs.open(config.PATH_CORPUS_TEST_RAW, 'r', 'utf-8') as ftest:
        test_reader = csv.reader(ftest)
        for row in test_reader:
            rows.append(row)
    p = Pool(70)
    for row in rows:
        p.apply_async(get_overlap_file, (row,), callback=callback)
    p.close()
    p.join()
    print('Get overlap result cost {}s.'.format(time.time()-start_time))
    tmp_time = time.time()
    with codecs.open(config.PATH_RESULT_OVERLAP_TOP20, 'r', 'utf-8') as f:
        with codecs.open(config.PATH_RESULT_OVERLAP_TOP20_SORTED, 'w', 'utf-8') as fw:
            # reader = tsv.TsvReader(f)
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
    print('Get overlap top20 result cost {}s.'.format(tmp_time-start_time))





