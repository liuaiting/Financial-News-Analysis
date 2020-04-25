# -*- coding: utf-8 -*-
"""
Created on Sat May 8 2018.
@author: Liu Aiting
@e-mail: liuaiting@bupt.edu.cn
"""
import time
import pickle

import config


def sort_result_file(path_result, path_result_sorted):
    start_time = time.time()
    print("Get sorted result for {}.".format(path_result))
    with open(path_result, 'r') as f:
        with open(path_result_sorted, 'wb') as fw:
            # fw.write('source_id\ttarget_id\tsimilarity\n')
            dic = dict()
            source_id_set = set()
            i = 0
            for row in f:
                row = row.strip().split()
                if i <= 5:
                    print(row)
                    i += 1
                source_id = int(row[0])
                target_id = int(row[1])
                similarity = float(row[2])
                if source_id not in source_id_set:
                    dic[source_id] = {}
                    source_id_set.add(source_id)
                dic[source_id][target_id] = similarity
                # rows.append([source_id, target_id, similarity])
            # results = sorted(rows, key=lambda x: (x[0], x[1]))
            pickle.dump(dic, fw)
            # for res in results:
            #     fw.write('\t'.join([str(r) for r in res]) + '\n')
    print('{} -> {} cost {}s.'.format(path_result, path_result_sorted, (time.time()-start_time)))


if __name__ == '__main__':
    # sort_result_file(config.PATH_RESULT_LSI, config.PATH_RESULT_LSI_SORTED)
    # sort_result_file(config.PATH_RESULT_DOC2VEC, config.PATH_RESULT_DOC2VEC_SORTED)
    # sort_result_file(config.PATH_RESULT_OVERLAP, config.PATH_RESULT_OVERLAP_SORTED)

    lsi_result = pickle.load(open(config.PATH_RESULT_LSI_SORTED, 'rb'))
    doc2vec_result = pickle.load(open(config.PATH_RESULT_DOC2VEC_SORTED, 'rb'))
    overlap_result = pickle.load(open(config.PATH_RESULT_OVERLAP_SORTED, 'rb'))

    test_ids = open(config.PATH_CORPUS_TEST_ID, 'r').readlines()
    test_ids = [int(i.strip()) for i in test_ids]

    fw = open(config.PATH_RESULT_FINAL, 'w')
    fw.write("source_id\ttarget_id\tsimilarity\n")
    for source_id in test_ids:
        results = []
        for target_id in range(1, 485686 + 1):
            print(source_id, target_id)
            lsi_sim = lsi_result[source_id][target_id]
            doc2vec_sim = doc2vec_result[source_id][target_id]
            overlap_sim = overlap_result[source_id][target_id]
            sim = lsi_sim * config.lsi_weight + doc2vec_sim * config.doc2vec_weight + overlap_sim * config.overlap_weight
            res = [int(source_id), int(target_id), float(sim)]
            results.append(res)
        top20 = sorted(results, key=lambda x: -x[2])[:21]
        num = 0
        for res in top20:
            if res[0] != res[1] and num <= 20:
                fw.write('\t'.join([str(r) for r in res]) + '\n')
                num += 1
    fw.close()




