import config
import sqlite3


def res_with_title(path, path_res):
    conn1 = sqlite3.connect(config.PATH_CORPUS_TRAIN_DB)
    c1 = conn1.cursor()

    with open(path, 'r') as f:
        with open(path_res, 'w') as fw:
            lines = f.readlines()[1:]
            for line in lines:
                row = line.strip().split('\t')
                source_id = row[0]
                target_id = row[1]
                similarity = row[2]
                c1.execute("SELECT title FROM KB WHERE id=?", (source_id,))
                source_title = c1.fetchone()[0]
                c1.execute("SELECT title FROM KB WHERE id=?", (target_id,))
                target_title = c1.fetchone()[0]
                # print(source_id, target_id)
                # print(source_title)
                # print(target_title)
                # print('\n')
                res = [source_id, target_id, similarity, source_title, target_title]
                fw.write('\t'.join([str(r) for r in res]) + '\n')
    conn1.close()


# res_with_title(config.PATH_RESULT_BM25_JIEBA_TOP20_SORTED, config.PATH_RESULT_BM25_JIEBA_TOP20_SORTED + '.title')
# res_with_title(config.PATH_RESULT_BM25_TOP20_SORTED, config.PATH_RESULT_BM25_TOP20_SORTED + '.title')
#

with open('./result/result_bm25_jieba_top20_sorted.txt') as f:
    with open('final_top20.txt', 'w') as fw:
        for line in f:
            fw.write('\t'.join(line.split('\t')[:2]) + '\n')
