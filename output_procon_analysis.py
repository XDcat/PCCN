# -*- coding:utf-8 -*-
'''
__author__ = 'XD'
__mtime__ = 2021/9/22
__project__ = Cov2_protein
Fix the Problem, Not the Blame.
'''

import json
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline, interp1d
import logging
import logconfig

import matplotlib.pyplot as plt

logconfig.setup_logging()
log = logging.getLogger("cov2")


def output_excel(analysis, outpath="./data/procon/analysis.xlsx"):
    # type1
    type1 = []
    type2 = []
    type3 = []
    keys = []
    for i, row in analysis.items():
        # k = "{}-{}-{} ({})".format(i, row["WHO label"], row["Lineage + additional mutations"], row["Spike mutations of interest"])
        # k = "编号:{}\nWHO 标签:{}\nLineage:{}\n变异:{}".format(i, row["WHO label"], row["Lineage + additional mutations"], row["Spike mutations of interest"])
        k = "编号:{}, WHO 标签:{}, Lineage:{}, 变异:{}, 类别:{}".format(i, row["WHO label"],
                                                                row["Lineage + additional mutations"],
                                                                row["Spike mutations of interest"], row["type"])
        keys.append(k)
        type1.append(pd.DataFrame(row["type1"]))
        type2.append(pd.DataFrame(row["type2"]))
        type3.append(pd.DataFrame(row["type3"]))

    type1 = pd.concat(type1, keys=keys)
    type1 = type1.reindex(["position", "rank", "information"], axis=1)
    count = 1612
    type1["rate(%)"] = type1["rank"] / count * 100

    type2 = pd.concat(type2, keys=keys)
    type2 = type2.reindex(["site1", "site2", "rank", "info"], axis=1)
    count = count ** 2
    type2["rate(%)"] = type2["rank"] / count * 100

    type3 = pd.concat(type3, keys=keys)
    if type3.shape[0] == 0:
        type3 = pd.Series({"msg": "没有结果"})
    log.info("type1 = %s", type1)
    log.info("type2 = %s", type2)
    log.info("type3 = %s", type3)

    # 写入文件
    log.info("输出文件: %s", outpath)
    with pd.ExcelWriter(outpath) as writer:
        type1.to_excel(writer, sheet_name="type1")
        type2.to_excel(writer, sheet_name="type2")
        type3.to_excel(writer, sheet_name="type3")


def two_degree_bc(x, y, dots_num=100):  # bezier curve
    """二阶贝塞尔曲线"""
    x1, x2, x3 = x
    y1, y2, y3 = y
    xt = []  # 目标点的x坐标
    yt = []  # 目标点的y坐标
    x_dots12 = np.linspace(x1, x2, dots_num)  # 线段AB的x坐标
    y_dots12 = np.linspace(y1, y2, dots_num)  # 线段AB的y坐标
    x_dots23 = np.linspace(x2, x3, dots_num)  # 线段BC的x坐标
    y_dots23 = np.linspace(y2, y3, dots_num)  # 线段BC的y坐标
    for i in range(dots_num):  # 获得目标点的轨迹
        x = x_dots12[i] + (x_dots23[i] - x_dots12[i]) * i / (dots_num - 1)
        y = y_dots12[i] + (y_dots23[i] - y_dots12[i]) * i / (dots_num - 1)
        xt.append(x)
        yt.append(y)
    xt = np.array(xt)
    yt = np.array(yt)
    return xt, yt


def output_picture(analysis, outpath="./data/procon/analysis.png", font_size="x-large"):
    type1 = []  # 存储 type1
    type2 = []
    for i, row in analysis.items():
        # type1
        # if str(row["WHO label"]) == "nan":
        #     continue
        t1 = row["type1"]
        t1 = pd.DataFrame(t1)

        t1["name"] = "{}. {}".format(i, row["WHO label"])
        t1["y"] = int(i)

        t1["idx"] = t1["position"].str[:-1].astype(int)
        position_2_aa = {int(i[1: -1]): i for i in row["aas"]}  # 位置的到变异的映射
        t1["aa"] = t1["idx"].map(position_2_aa)

        # type2
        t2 = row["type2"]
        t2 = pd.DataFrame(t2)
        t2["y1"] = t2["y2"] = int(i)

        type1.append(t1)
        type2.append(t2)

    type1 = pd.concat(type1)
    type2 = pd.concat(type2)

    # 建立位点到x坐标的映射
    pst_2_x = type1["idx"].drop_duplicates().sort_values().to_list()
    pst_2_x = {j: i for i, j in enumerate(pst_2_x)}

    type1["x"] = type1["idx"].map(pst_2_x)
    type2["x1"] = type2["site1"].str[:-1].astype(int).map(pst_2_x)
    type2["x2"] = type2["site2"].str[:-1].astype(int).map(pst_2_x)

    # 构建 type2 曲线最高点
    gap = 1
    type2["x3"] = (type2["x1"] + type2["x2"]) / 2
    log.debug("type2 = %s", type2)
    type2["y3"] = (type2["y1"] + gap / 4) + (type2["x2"] - type2["x1"]).abs() / (len(pst_2_x) - 2)

    log.debug("type1 = %s", type1)
    log.debug("type2 = %s", type2)

    # 绘图
    flg_size = (len(pst_2_x), len(analysis))
    fig_size = np.array(flg_size)
    flg, ax = plt.subplots(figsize=fig_size.tolist())
    # flg, ax = plt.subplots()
    ax.scatter(type1["x"], type1["y"], s=0)

    # 添加文字：变异
    for i, row in type1.iterrows():
        x, y, txt = row["x"], row["y"], row["aa"]
        ax.text(x, y, txt, ha="center", va="center", size=font_size)

    # 绘制曲线

    for i, row in type2.iterrows():
        x1, x2, x3, y1, y2, y3 = row[["x1", "x2", "x3", "y1", "y2", "y3"]]
        x = [x1, x3, x2]
        y = [y1, y3, y2]
        x_smooth, y_smooth = two_degree_bc(x, y)
        y_smooth += 0.15
        ax.plot(x_smooth, y_smooth)

    # 添加坐标标签
    xtick_names = type1["position"].drop_duplicates()
    arg_sort = xtick_names.str[:-1].astype(int).argsort()
    xtick_names = xtick_names.iloc[arg_sort]
    ytick_names = type1["name"].drop_duplicates()
    arg_sort = ytick_names.apply(lambda x: int(x.split(".")[0])).astype(int).argsort()
    ytick_names = ytick_names.iloc[arg_sort]

    ax.set_xticks(range(len(xtick_names)))
    ax.set_xticklabels(xtick_names, rotation=0, size=font_size)
    ax.set_yticks(range(len(ytick_names)))
    ax.set_yticklabels(ytick_names, size=font_size)

    # ax.invert_yaxis()  # 反转y轴
    ax.set_ymargin(0.03)

    flg.show()
    flg.savefig(outpath, dpi=300)


if __name__ == '__main__':
    # 加载数据
    analysis_file = "./data/procon/analysis.json"
    with open(analysis_file) as f:
        analysis = json.load(f)
    log.debug("analysis = %s", analysis)

    # 解析到excel
    # output_excel(analysis)
    output_picture(analysis)
