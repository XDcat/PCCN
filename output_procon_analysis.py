import json
import math
import seaborn as sns
from Bio import SeqIO
import networkx as nx

import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline, interp1d
from pyecharts import options as opts
from pyecharts.charts import Graph
import logging
import logconfig

import matplotlib.pyplot as plt

# logconfig.setup_logging()
log = logging.getLogger("cov2")


def output_excel(analysis, outpath="./data/procon/analysis.xlsx"):
    # type1
    type1 = []
    type2 = []
    type3 = []
    keys = []
    for i, row in analysis.items():
        # k = "{}-{}-{} ({})".format(i, row["WHO label"], row["Lineage + additional mutations"], row["Spike mutations of interest"])
        # k = "index:{}\nWHO tag:{}\nLineage:{}\n变异:{}".format(i, row["WHO label"], row["Lineage + additional mutations"], row["Spike mutations of interest"])
        k = "index:{}, WHO tag:{}, Lineage:{}, mutation:{}, type:{}".format(i, row["WHO label"],
                                                                            row["Lineage + additional mutations"],
                                                                            row["Spike mutations of interest"],
                                                                            row["type"])
        if row["WHO label"]:
            k = "{}-{}({})".format(i, row["Lineage + additional mutations"], row["WHO label"], )
        else:
            k = "{}-{}".format(i, row["Lineage + additional mutations"], )

        keys.append(k)
        type1.append(pd.DataFrame(row["type1"]))
        type2.append(pd.DataFrame(row["type2"]))
        type3.append(pd.DataFrame(row["type3"]))

    count = 1612
    type1 = pd.concat(type1, keys=keys)
    type1 = type1.reindex(["aa", "position", "rank", "information"], axis=1)
    type1["rate(%)"] = type1["rank"] / count * 100

    type2 = pd.concat(type2, keys=keys)
    type2 = type2.reindex(["site1", "site2", "rank", "info"], axis=1)
    count = count ** 2
    type2["rate(%)"] = type2["rank"] / count * 100

    type3 = pd.concat(type3, keys=keys)
    if type3.shape[0] == 0:
        type3 = pd.Series({"msg": "no result"})
    log.info("type1 = %s", type1)
    log.info("type2 = %s", type2)
    log.info("type3 = %s", type3)

    # output
    log.info("path: %s", outpath)
    with pd.ExcelWriter(outpath) as writer:
        type1.to_excel(writer, sheet_name="type1")
        type2.to_excel(writer, sheet_name="type2")
        type3.to_excel(writer, sheet_name="type3")


def plot_2D(type1, type2, pst_2_x, analysis, outpath, font_size):
    def two_degree_bc(x, y, dots_num=100):  # bezier curve
        """bezier curve"""
        x1, x2, x3 = x
        y1, y2, y3 = y
        xt = []  # x
        yt = []  # y
        x_dots12 = np.linspace(x1, x2, dots_num)  # x for AB
        y_dots12 = np.linspace(y1, y2, dots_num)  # y for AB
        x_dots23 = np.linspace(x2, x3, dots_num)  # x for BC
        y_dots23 = np.linspace(y2, y3, dots_num)  # y for BC
        for i in range(dots_num):
            x = x_dots12[i] + (x_dots23[i] - x_dots12[i]) * i / (dots_num - 1)
            y = y_dots12[i] + (y_dots23[i] - y_dots12[i]) * i / (dots_num - 1)
            xt.append(x)
            yt.append(y)
        xt = np.array(xt)
        yt = np.array(yt)
        return xt, yt

    flg_size = (len(pst_2_x), len(analysis))
    fig_size = np.array(flg_size)
    flg, ax = plt.subplots(figsize=fig_size.tolist())
    ax.scatter(type1["x"], type1["y"], s=0)

    # add text
    for i, row in type1.iterrows():
        x, y, txt = row["x"], row["y"], row["aa"]
        ax.text(x, y, txt, ha="center", va="center", size=font_size)

    # draw curve
    for i, row in type2.iterrows():
        x1, x2, x3, y1, y2, y3 = row[["x1", "x2", "x3", "y1", "y2", "y3"]]
        x = [x1, x3, x2]
        y = [y1, y3, y2]
        x_smooth, y_smooth = two_degree_bc(x, y)
        y_smooth += 0.15
        alpha = row["rate"] / 100
        ax.plot(x_smooth, y_smooth, "C1", alpha=alpha)
        # ax.plot(x_smooth, y_smooth, )

    # add axis tag
    xtick_names = type1["position"].drop_duplicates()
    arg_sort = xtick_names.str[:-1].astype(int).argsort()
    xtick_names = xtick_names.iloc[arg_sort]
    ytick_names = type1["name"].drop_duplicates()
    arg_sort = ytick_names.apply(lambda x: int(x.split("-")[0])).astype(int).argsort()
    ytick_names = ytick_names.iloc[arg_sort]

    ax.set_xticks(range(len(xtick_names)))
    ax.set_yticks(range(len(ytick_names)))
    ax.set_xticklabels(xtick_names, size=font_size)
    ax.set_yticklabels(ytick_names, size=font_size)
    ax.set_ymargin(0.03)

    # save
    flg.tight_layout()
    flg.show()
    flg.savefig(outpath, dpi=200)


def plot_graph(nodes, links, name, layout="circular"):
    c = (
        Graph(init_opts=opts.InitOpts(width="100%", height="1000px"))
        .add("", nodes, links, repulsion=8000, layout=layout, )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="count"),
            toolbox_opts=opts.ToolboxOpts(
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(pixel_ratio=3, background_color="white"))), )
        .render(name)
    )


def output_picture(analysis, outpath="./data/procon/analysis.png", font_size="x-large"):
    type2 = []
    type1 = []
    for i, row in analysis.items():
        # t1 = pd.DataFrame({"aa": row["aas"]})
        t1 = pd.DataFrame(row["type1"])
        t1["y"] = int(i) - 1
        # find name
        if row["WHO label"] != "nan":
            name = "{}-{}({})".format(i, row["Lineage + additional mutations"], row["WHO label"], )
        else:
            name = "{}-{}".format(i, row["Lineage + additional mutations"], )
        t1["name"] = name

        # type2
        t2 = row["type2"]
        t2 = pd.DataFrame(t2)
        t2["y1"] = t2["y2"] = int(i) - 1  # y

        type1.append(t1)
        type2.append(t2)

    type2 = pd.concat(type2)
    type1 = pd.concat(type1)

    type1["idx"] = type1["aa"].str[1:-1].astype(int)
    # type1["position"] = type1["aa"].str[1:-1] + type1["aa"].str[0]
    # map position to x, to find all mutations
    pst_2_x = type1["idx"].drop_duplicates().sort_values().to_list()
    pst_2_x = {j: i for i, j in enumerate(pst_2_x)}

    # x for type1 and type2
    type1["x"] = type1["idx"].map(pst_2_x)
    type2["x1"] = type2["site1"].str[:-1].astype(int).map(pst_2_x)
    type2["x2"] = type2["site2"].str[:-1].astype(int).map(pst_2_x)

    # generate top high dot for type2
    gap = 1
    type2["x3"] = (type2["x1"] + type2["x2"]) / 2
    log.debug("type2 = %s", type2)
    type2["y3"] = (type2["y1"] + gap / 4) + (type2["x2"] - type2["x1"]).abs() / (len(pst_2_x) - 2)

    log.debug("type1 = %s", type1)
    log.debug("type2 = %s", type2)

    # edge
    # type2_count = type2.loc[:, ["site1", "site2"]].value_counts().reset_index().rename({0: "count"}, axis=1)
    type2_info = type2.loc[:, ["site1", "site2", "info", "rate"]].drop_duplicates(["site1", "site2"])
    # log.debug("type2_count = %s", type2_count)
    log.debug("type2_info = %s", type2_info)
    # draw
    # 2D
    plot_2D(type1, type2, pst_2_x, analysis, outpath, font_size)
    # relationship
    nodes = type1.loc[:, ["position", "information"]]
    nodes = nodes.loc[:, "position"].value_counts()
    nodes = nodes.reset_index()
    nodes.columns = ["name", "symbolSize"]
    # nodes = nodes.drop_duplicates()
    log.debug("nodes = %s", nodes)
    nodes = nodes.to_dict("records")

    links = type2_info.loc[:, ["site1", "site2"]]
    links.columns = ["source", "target"]
    log.debug("links = %s", links)
    links = links.to_dict("records")

    plot_graph(nodes, links, "./data/procon/count.html", None)


def output_picture_global(fasta, analysis, parse1="./data/procon/type1_parse.csv",
                          parse2="./data/procon/type2_parse.csv"):
    type1 = pd.read_csv(parse1)
    type2 = pd.read_csv(parse2)
    # node
    nodes = ["{}{}".format(i + 1, j) for i, j in enumerate(fasta)]
    nodes = pd.DataFrame({"position": nodes})
    nodes = pd.merge(nodes, type1, how="left", left_on="position", right_on="position")
    nodes = nodes.loc[:, ["position", "information"]]
    nodes.columns = ["name", "symbolSize"]
    nodes = nodes.fillna(0)
    log.debug("nodes = %s", nodes)
    type2 = type2[type2["info"] > 300]
    log.debug("type2 = %s", type2)
    # edge
    links = type2.loc[:, ["site1", "site2"]]
    links.columns = ["source", "target"]
    log.debug("links = %s", links)

    # nodes = nodes.to_dict("records")
    # links = links.to_dict("records")
    # plot_graph(nodes, links, "./data/procon/global_info.html", "force")
    # draw relationship
    G = nx.Graph()
    for i, row in nodes.iterrows():
        G.add_node(row["name"], name=nodes["name"], )

    for i, row in links.iterrows():
        G.add_edge(row["source"], row["target"])

    log.debug("G.nodes = %s", G.nodes)
    log.debug("G.edges = %s", G.edges)
    log.debug("count nodes: %s", len(G.nodes))
    log.debug("count edge: %s", len(G.edges))
    # 1. all nodes
    flg = plt.figure(figsize=(20, 20))
    nx.draw_circular(G, node_size=1, width=1, with_labels=False)
    nx.draw_networkx_labels(G, nx.drawing.circular_layout(G, scale=1.05), font_size=3, font_color="g", )
    flg.show()
    flg.savefig("./data/procon/1. global_graph.png", dpi=300)

    # 2. delete all single node
    G_not_isolaties = G.subgraph([n for n in G if n not in list(nx.isolates(G))])
    flg = plt.figure(figsize=(20, 20))
    nx.draw_kamada_kawai(G_not_isolaties, node_size=5, width=1, with_labels=True)
    flg.show()
    flg.savefig("./data/procon/2. global_graph(remove isolated).png", dpi=300)

    # 3. all nodes, but display part tags
    psts = get_position_from_parse(analysis)
    G_psts = G.subgraph(psts)
    edge_colors = ["b" if (s1 in psts or s2 in psts) else "k" for s1, s2 in G.edges()]
    node_colors = ["r" if n in psts else "#1f78b4" for n in G.nodes]
    flg = plt.figure(figsize=(20, 20))
    nx.draw_circular(G, node_size=1, width=1, with_labels=False, edge_color=edge_colors, node_color=node_colors)
    nx.draw_networkx_labels(G_psts, nx.drawing.circular_layout(G, scale=1.01), font_size=3, font_color="r", )
    flg.show()
    flg.savefig("./data/procon/3. global_graph(only display part labels).png", dpi=300)


def get_position_from_parse(analysis):
    aas = []
    for i, row in analysis.items():
        aas += row["aas"]

    aas = pd.Series(aas)
    aas = aas.str[1:-1] + aas.str[0]
    aas = aas.drop_duplicates()
    aas = aas.to_list()
    log.debug("aas = %s", aas)
    log.debug("len(aas) = %s", len(aas))
    return aas


if __name__ == '__main__':
    # load data
    analysis_file = "./data/procon/analysis.json"
    with open(analysis_file) as f:
        analysis = json.load(f)
    log.debug("analysis = %s", analysis)

    fasta = SeqIO.parse(r"./data/YP_009724390.1.txt", "fasta")
    fasta = next(fasta).seq

    # output to excel
    # output_excel(analysis)
    # draw
    output_picture(analysis)
    # draw global figure
    # output_picture_global(fasta, analysis)
