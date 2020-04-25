"""
Created on Sat May 8, 2018.
@author: Liu Aiting
@email: liuaiting@bupt.edu.cn
"""
import codecs
import re
import sqlite3
import time
import csv
import tsv
import jieba
from contextlib import closing

import config


def lower_case(line):
    return str(line).lower()


def tokenizer(sentence):
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


def rewrite(line):
    line = lower_case(line)
    line = tokenizer(line)
    line = ' '.join(line)
    return line


def read_file(path_file):
    with codecs.open(path_file, 'r', 'utf-8', errors='ignore') as f:
        return f.readlines()


def build_db(path_in, path_out):
    start_time = time.time()

    # Create a db if there not exists one.
    conn = sqlite3.connect(path_out)
    print("Opened database successfully")

    c = conn.cursor()

    # Create a table KB to save KB triples.
    c.execute("""CREATE TABLE KB
    (id INT PRIMARY KEY NOT NULL,
    title TEXT NOT NULL);""")
    print("Table created successfully")

    # Insert KB triples into KB table.
    with codecs.open(path_in, 'r', 'utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            _id = row[0]
            _title = row[1]
            tem = (_id, _title,)
            c.execute("INSERT INTO KB(id,title) \
                VALUES(?,?)", tem)

    conn.commit()
    print("Records created successfully")
    conn.close()

    end_time = time.time()
    print("Created %s cost %.6f s." % (path_out, end_time - start_time))


def get_title(idx, path_tain_db=config.PATH_CORPUS_TRAIN_DB):
    with closing(sqlite3.connect(config.path_train_db)) as conn1:
        c1 = conn1.cursor()
        c1.execute("SELECT title FROM KB WHERE id=?", (idx,))
        results = c1.fetchall()
        return results






