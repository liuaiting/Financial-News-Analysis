# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import codecs

df = pd.read_csv("shortnews_dataframe.csv", sep="|", encoding="utf-8")
# print(df.head(40))
new_df = df.sort_values(by="post_time")
print(new_df.head(40))
new_df.to_csv("data.csv", sep="|", encoding="utf-8", index=False)
# with codecs.open("data/shortnews_dataframe.csv", "r", "utf-8") as f:
#     with codecs.open("data/test_x.csv", "w", "utf-8") as fw:
#         f.readline()
#         # a = "31|2018-03-01 03:41:57|瑞信集团全球策略师Andrew Garthwaite指出，在美国利率上升的背景下，科技股看上去像是天堂。"
#         # print(a.split("|"))
#         dataset = dict()
#         last_id = 0
#         for line in f:
#             cols = line.split("|")
#             if len(cols) != 3:
#                 print(last_id)
#                 dataset[last_id]["title"] += line.strip()
#             else:
#                 print(last_id)
#                 dataset[cols[0]] = {"post_time": cols[1],
#                                          "title": str(cols[2]).strip()}
#                 last_id = cols[0]
#
#         for idx, content in dataset.items():
#             post_time = content["post_time"]
#             title = content["title"]
#             # print(idx, post_time, title)
#             fw.write(str(idx) + "|" + str(post_time) + "|" + str(title) + "\n")
#
# df = pd.read_csv("data/shortnews_dataframe.csv", sep="|", encoding="utf-8", header=None)
# df = pd.read_csv("data/price_per_day.csv", sep=",", header="infer",
#                  usecols=["trade_time", "open_price", "close_price"])#, "high_price", "low_price"])
# df["cha"] = df.apply(lambda row: row["close_price"] - row["open_price"], axis=1)
# print(df.head())
# plt.figure()
# df.plot(x="trade_time", y="cha")
#
# plt.show()
