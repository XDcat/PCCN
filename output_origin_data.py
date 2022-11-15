# -*- coding:utf-8 -*-
import json
import math
from Bio import SeqIO

import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline, interp1d
from pyecharts import options as opts
from pyecharts.charts import Graph
import logging
import logconfig

import matplotlib.pyplot as plt

logconfig.setup_logging()
log = logging.getLogger("cov2")
if __name__ == '__main__':
    analysis_file = "./data/procon/analysis.json"
    parse1 = "./data/procon/type1_parse.csv"
    type1 = pd.read_csv(parse1)
    with open(analysis_file) as f:
        analysis = json.load(f)

    all_exit_aa = []
    for i, item in analysis.items():
        all_exit_aa += item["aas"]

    # count occurrence number
    count_aa = pd.value_counts(all_exit_aa)
    log.debug("count_aa = %s", count_aa)
    flg, ax = plt.subplots()
    count_aa.plot(ax=ax)
    flg.savefig("./data/procon/count aas.png")

    # procon
    data = pd.value_counts([i[1:-1] + i[0] for i in all_exit_aa])
    data = data.to_frame("count")
    # data = data.reset_index().rename({"index": "aa"}, axis=1)
    data["position"] = data.index
    data = pd.merge(data, type1, left_on="position", right_on="position", how="left")
    data["rate"] = data["rate"].fillna(100)
    data["x"] = data["rate"]
    data["y"] = data["count"]
    log.debug("data = %s", data)

    # scatter
    flg, ax = plt.subplots(figsize=(10, 5))
    x = data["x"].to_list()
    y = data["y"].to_list()
    # ax.spines["left"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    ax.scatter(x, y, c="g", s=np.ones(len(x)) * 15)
    # ax.set_xlim(10, 100)
    ax.set_ylim(0, )
    ax.set_xlabel("rank(%)")
    ax.set_ylabel("count")
    # text
    for i, row in data.iterrows():
        x, y, txt = row["x"], row["y"], row["position"]
        ax.text(x + 1, y, txt, fontsize="xx-small", )
    flg.savefig("./data/procon/count and rank.png", dpi=300)
