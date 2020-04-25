import os

# num topics for LSI model
num_topics = 5000
# corpus
PATH_CORPUS_ROOT = './path_corpus/'
PATH_CORPUS_TRAIN_RAW = PATH_CORPUS_ROOT + 'train_data.csv'
PATH_CORPUS_TRAIN = PATH_CORPUS_ROOT + 'corpus_train.txt'
PATH_CORPUS_TRAIN_CUT = PATH_CORPUS_ROOT + 'train_data_cut.txt'
PATH_CORPUS_TRAIN_DB = PATH_CORPUS_ROOT + 'train_data.db'
PATH_CORPUS_TRAIN_CUT_ID = PATH_CORPUS_ROOT + 'train_data_cut_id.txt'
PATH_CORPUS_TRAIN_DOC2VEC = PATH_CORPUS_ROOT + 'train_doc2vec.txt'
PATH_CORPUS_TRAIN_BM25 = PATH_CORPUS_ROOT + 'train_bm25.txt'
PATH_CORPUS_TRAIN_BM25_THU = PATH_CORPUS_ROOT + 'train_bm25_thu.txt'
PATH_CORPUS_TRAIN_BM25_NLPIR = PATH_CORPUS_ROOT + 'train_bm25_NLPIR.txt'
PATH_CORPUS_TRAIN_BM25_JIEBA = PATH_CORPUS_ROOT + 'train_bm25_jieba.txt'

PATH_CORPUS_TEST_RAW = PATH_CORPUS_ROOT + 'test_data.csv'
PATH_CORPUS_TEST = PATH_CORPUS_ROOT + 'corpus_test.txt'
PATH_CORPUS_TEST_CUT = PATH_CORPUS_ROOT + 'test_data_cut.txt'
PATH_CORPUS_TEST_CUT_ID = PATH_CORPUS_ROOT + 'test_data_cut_id.txt'
PATH_CORPUS_TEST_DOC2VEC = PATH_CORPUS_ROOT + 'test_doc2vec.txt'
PATH_CORPUS_TEST_ID = PATH_CORPUS_ROOT + 'test_data_id.txt'
PATH_CORPUS_TEST_BM25 = PATH_CORPUS_ROOT + 'test_bm25.txt'
PATH_CORPUS_TEST_BM25_THU = PATH_CORPUS_ROOT + 'test_bm25_thu.txt'
PATH_CORPUS_TEST_BM25_NLPIR = PATH_CORPUS_ROOT + 'test_bm25_NLPIR.txt'
PATH_CORPUS_TEST_BM25_JIEBA = PATH_CORPUS_ROOT + 'test_bm25_jieba.txt'

# stop words
PATH_STOP_WORDS = PATH_CORPUS_ROOT + 'stop_words.txt'
# word2vec and vocab
PATH_WORD2VEC = PATH_CORPUS_ROOT + 'vec.txt'
PATH_VOCAB = PATH_CORPUS_ROOT + 'vocab.txt'
# model
PATH_MODEL_ROOT = "./path_model/"
# PATH_TF_IDF_MODEL = PATH_MODEL_ROOT + "tf_idf.model"
# PATH_LSI_TF_IDF_MODEL = PATH_MODEL_ROOT + "lsi_tf_idf.model"
# PATH_LDA_TF_IDF_MODEL = PATH_MODEL_ROOT + "lda_tf_idf_model"
PATH_LSI_MODEL = PATH_MODEL_ROOT + "lsi_{}.model".format(str(num_topics))
# PATH_LDA_MODEL = PATH_MODEL_ROOT + "lda_{}.model".format(str(num_topics))
PATH_LSI_INDEX_MODEL = PATH_MODEL_ROOT + "lsi_index_{}.model".format(str(num_topics))
PATH_WORDS_DIC = PATH_MODEL_ROOT + "words.dic"
PATH_MM_CORPUS = PATH_MODEL_ROOT + "corpus.mm"

PATH_SIMHASH_MODEL_D = PATH_MODEL_ROOT + 'simhash_D.pk'
PATH_SIMHASH_MODEL_VOC = PATH_MODEL_ROOT + 'simhash_voc.pk'

PATH_BM25_MODEL = PATH_MODEL_ROOT + 'bm25_model.pk'
PATH_BM25_MODEL_AVGIDF = PATH_MODEL_ROOT + 'bm25_avg_idf.pk'
PATH_BM25_CORPUS = PATH_MODEL_ROOT + 'bm25_corpus.pk'
PATH_BM25_DICTIONARY = PATH_MODEL_ROOT + 'bm25_dictionary.pk'

PATH_BM25_PRE_MODEL = PATH_MODEL_ROOT + 'bm25_pre_model.pk'
PATH_BM25_PRE_MODEL_AVGIDF = PATH_MODEL_ROOT + 'bm25_pre_avg_idf.pk'
PATH_BM25_PRE_CORPUS = PATH_MODEL_ROOT + 'bm25_pre_corpus.pk'
PATH_BM25_PRE_DICTIONARY = PATH_MODEL_ROOT + 'bm25_pre_dictionary.pk'

PATH_BM25_THU_MODEL = PATH_MODEL_ROOT + 'bm25_thu_model.pk'
PATH_BM25_THU_MODEL_AVGIDF = PATH_MODEL_ROOT + 'bm25_thu_avg_idf.pk'
PATH_BM25_THU_CORPUS = PATH_MODEL_ROOT + 'bm25_thu_corpus.pk'
PATH_BM25_THU_DICTIONARY = PATH_MODEL_ROOT + 'bm25_thu_dictionary.pk'

PATH_BM25_NLPIR_MODEL = PATH_MODEL_ROOT + 'bm25_NLPIR_model.pk'
PATH_BM25_NLPIR_MODEL_AVGIDF = PATH_MODEL_ROOT + 'bm25_NLPIR_avg_idf.pk'
PATH_BM25_NLPIR_CORPUS = PATH_MODEL_ROOT + 'bm25_NLPIR_corpus.pk'
PATH_BM25_NLPIR_DICTIONARY = PATH_MODEL_ROOT + 'bm25_NLPIR_dictionary.pk'

PATH_BM25_JIEBA_MODEL = PATH_MODEL_ROOT + 'bm25_jieba_model.pk'
PATH_BM25_JIEBA_MODEL_AVGIDF = PATH_MODEL_ROOT + 'bm25_jieba_avg_idf.pk'
PATH_BM25_JIEBA_CORPUS = PATH_MODEL_ROOT + 'bm25_jieba_corpus.pk'
PATH_BM25_JIEBA_DICTIONARY = PATH_MODEL_ROOT + 'bm25_jieba_dictionary.pk'
# result
PATH_RESULT_ROOT = './result/'
PATH_RESULT_LSI = PATH_RESULT_ROOT + "result_{}.txt".format(str(num_topics))
PATH_RESULT_LSI_SORTED = PATH_RESULT_ROOT + "result_{}_sorted.pk".format(str(num_topics))
PATH_RESULT_LSI_TOP20 = PATH_RESULT_ROOT + "result_{}_top20.txt".format(str(num_topics))
PATH_RESULT_LSI_TOP20_SORTED = PATH_RESULT_ROOT + "result_{}_top20_sorted.txt".format(str(num_topics))
# result of sequence overlap model
PATH_RESULT_OVERLAP = PATH_RESULT_ROOT + 'result_overlap.txt'
PATH_RESULT_OVERLAP_SORTED = PATH_RESULT_ROOT + 'result_overlap_sorted.pk'
PATH_RESULT_OVERLAP_TOP20 = PATH_RESULT_ROOT + 'result_overlap_top20.txt'
PATH_RESULT_OVERLAP_TOP20_SORTED = PATH_RESULT_ROOT + 'result_overlap_top20_sorted.txt'
# result of average word2vec model
PATH_RESULT_DOC2VEC = PATH_RESULT_ROOT + 'result_doc2vec.txt'
PATH_RESULT_DOC2VEC_SORTED = PATH_RESULT_ROOT + 'result_doc2vec_sorted.pk'
PATH_RESULT_DOC2VEC_TOP20 = PATH_RESULT_ROOT + 'result_doc2vec_top20.txt'
PATH_RESULT_DOC2VEC_TOP20_SORTED = PATH_RESULT_ROOT + 'result_doc2vec_top20_sorted.txt'
# result of simhash model
PATH_RESULT_SIMHASH = PATH_RESULT_ROOT + 'result_simhash.txt'
PATH_RESULT_SIMHASH_SORTED = PATH_RESULT_ROOT + 'result_simhash_sorted.pk'
PATH_RESULT_SIMHASH_TOP20 = PATH_RESULT_ROOT + 'result_simhash_top20.txt'
PATH_RESULT_SIMHASH_TOP20_SORTED = PATH_RESULT_ROOT + 'result_simhash_top20_sorted.txt'
# result of bm25 model (jieba, jieba.cut(cut_all=False))
PATH_RESULT_BM25 = PATH_RESULT_ROOT + 'result_bm25.txt'
PATH_RESULT_BM25_SORTED = PATH_RESULT_ROOT + 'result_bm25_sorted.pk'
PATH_RESULT_BM25_TOP20 = PATH_RESULT_ROOT + 'result_bm25_top20.txt'
PATH_RESULT_BM25_TOP20_SORTED = PATH_RESULT_ROOT + 'result_bm25_top20_sorted.txt'
# result of bm25 model (jieba, jieba.cut(cut_all=False), delete stop words)
PATH_RESULT_BM25_PRE = PATH_RESULT_ROOT + 'result_bm25_pre.txt'
PATH_RESULT_BM25_PRE_SORTED = PATH_RESULT_ROOT + 'result_bm25_pre_sorted.pk'
PATH_RESULT_BM25_PRE_TOP20 = PATH_RESULT_ROOT + 'result_bm25_pre_top20.txt'
PATH_RESULT_BM25_PRE_TOP20_SORTED = PATH_RESULT_ROOT + 'result_bm25_pre_top20_sorted.txt'
# result of bm25 model (thulac))
PATH_RESULT_BM25_THU = PATH_RESULT_ROOT + 'result_bm25_thu.txt'
PATH_RESULT_BM25_THU_SORTED = PATH_RESULT_ROOT + 'result_bm25_thu_sorted.pk'
PATH_RESULT_BM25_THU_TOP20 = PATH_RESULT_ROOT + 'result_bm25_thu_top20.txt'
PATH_RESULT_BM25_THU_TOP20_SORTED = PATH_RESULT_ROOT + 'result_bm25_thu_top20_sorted.txt'
# result of bm25 model (NLPIR)
PATH_RESULT_BM25_NLPIR = PATH_RESULT_ROOT + 'result_bm25_NLPIR.txt'
PATH_RESULT_BM25_NLPIR_SORTED = PATH_RESULT_ROOT + 'result_bm25_NLPIR_sorted.pk'
PATH_RESULT_BM25_NLPIR_TOP20 = PATH_RESULT_ROOT + 'result_bm25_NLPIR_top20.txt'
PATH_RESULT_BM25_NLPIR_TOP20_SORTED = PATH_RESULT_ROOT + 'result_bm25_NLPIR_top20_sorted.txt'
# result of bm25 model (jieba, jieba.cut_for_research())
PATH_RESULT_BM25_JIEBA = PATH_RESULT_ROOT + 'result_bm25_jieba.txt'
PATH_RESULT_BM25_JIEBA_SORTED = PATH_RESULT_ROOT + 'result_bm25_jieba_sorted.pk'
PATH_RESULT_BM25_JIEBA_TOP20 = PATH_RESULT_ROOT + 'result_bm25_jieba_top20.txt'
PATH_RESULT_BM25_JIEBA_TOP20_SORTED = PATH_RESULT_ROOT + 'result_bm25_jieba_top20_sorted.txt'

lsi_weight = 0.4
doc2vec_weight = 0.35
overlap_weight = 0.25
PATH_RESULT_FINAL = PATH_RESULT_ROOT + 'result_top20_{}_{}_{}.txt'.format(lsi_weight, doc2vec_weight, overlap_weight)

if not os.path.exists(PATH_CORPUS_ROOT):
    os.makedirs(PATH_CORPUS_ROOT)
if not os.path.exists(PATH_MODEL_ROOT):
    os.makedirs(PATH_MODEL_ROOT)
if not os.path.exists(PATH_RESULT_ROOT):
    os.makedirs(PATH_RESULT_ROOT)
