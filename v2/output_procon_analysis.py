import json
import math
import seaborn as sns
import time
from Bio import SeqIO
import networkx as nx
from networkx.algorithms.centrality import degree_centrality, betweenness_centrality, closeness_centrality
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import comb
from itertools import combinations, permutations
from collections import defaultdict
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
from sklearn import preprocessing
# 日志
import logging
import logconfig
from typing import List

logconfig.setup_logging()
log = logging.getLogger("cov2")

AA = ['-', 'C', 'F', 'I', 'L', 'M', 'V', 'W', 'Y', 'A', 'T', 'D', 'E', 'G', 'P', 'N', 'Q', 'S', 'H', 'R', 'K']


class AnalysisMutationGroup:
    def __init__(self, analysis="../data/procon/analysis.json"):
        # 关注的变异组的相关数据
        with open(analysis) as f:
            self.analysis: dict = json.load(f)

        self.aa_groups = self.get_aa_groups()
        self.non_duplicated_aas = self.get_non_duplicated_aas()

    def get_aa_groups(self):
        aas = []
        for i, row in self.analysis.items():
            aas.append(row["aas"])
        return aas

    def get_non_duplicated_aas(self):
        aas = []
        for i, row in self.analysis.items():
            aas += row["aas"]
        aas = list(set(aas))
        return aas


class ProConNetwork:
    def __init__(self,
                 data_dir="../data/procon",
                 parse1="../data/procon/type1_parse.csv",
                 parse2="../data/procon/type2_parse.csv",
                 fasta_file="../data/YP_009724390.1.txt",
                 ):
        self.data_dir = data_dir

        log.info("构造图...")
        # procon 计算的所有的结果
        self.type1 = pd.read_csv(parse1)  # 单点
        self.type2 = pd.read_csv(parse2)  # 成对
        # 归一化: 单点
        self.type1["info_norm"] = self._normalize_info(self.type1["information"].values)
        self.type2["info_norm"] = self._normalize_info(self.type2["info"].values)

        # fasta 序列
        self.fasta = next(SeqIO.parse(fasta_file, "fasta")).seq

        # 节点
        self.nodes = self._get_nodes(self.fasta, self.type1)
        # 边
        self.links = self._get_links(self.type2)
        # 构成图
        self.G = self._get_G(self.links, self.nodes)

        # 中心性
        log.info("确定中心性...")
        self.degree_c, self.betweenness_c, self.closeness_c = self._get_centralities()

    @staticmethod
    def _normalize_info(info: np.ndarray):
        info = info.reshape(-1, 1)
        mms = preprocessing.MinMaxScaler().fit(info)
        return mms.transform(info)

    @staticmethod
    def _aa2position(aa: str):
        if aa[0] in AA and aa[-1] in aa and aa[1:-1].isdigit():
            return aa[1:-1] + aa[0]
        elif aa[:-1].isdigit() and aa[-1] in aa:
            return aa
        else:
            raise RuntimeError(f"无法转化aa={aa}")

    def _get_nodes(self, fasta, type1):
        nodes = ["{}{}".format(i + 1, j) for i, j in enumerate(fasta)]
        nodes = pd.DataFrame({"position": nodes})
        nodes = pd.merge(nodes, type1, how="left", left_on="position", right_on="position")
        nodes = nodes.loc[:, ["position", "info_norm"]]
        nodes.columns = ["name", "size"]
        nodes = nodes.fillna(0)
        return nodes

    def _get_links(self, type2, ):
        links = type2.loc[:, ["site1", "site2", "info_norm"]]
        links.columns = ["source", "target", "weight"]
        return links

    def _get_G(self, links, nodes):
        # 绘制关系图
        G = nx.Graph()
        for i, row in nodes.iterrows():
            G.add_node(row["name"], size=row["size"])

        for i, row in links.iterrows():
            G.add_edge(row["source"], row["target"], weight=row["weight"])
        return G

    def display(self):
        log.debug("self.type1 = %s", self.type1)
        log.debug("self.type2 = %s", self.type2)
        log.info("nodes = %s", self.nodes)
        log.info("links = %s", self.links)
        log.debug("dc = %s", self.degree_c)
        log.debug("cc = %s", self.closeness_c)
        log.debug("bc = %s", self.betweenness_c)

    @staticmethod
    def count_distribution(d: dict):
        values = list(d.values())
        counts = pd.value_counts(values)
        counts = counts.sort_index()
        counts = counts.to_dict()
        return counts.keys(), counts.values()

    def _get_centralities(self):
        # 点度中心性 degree
        outpath = [os.path.join(self.data_dir, "cache", f"{i}_centrality.json") for i in
                   ["degree", "betweenness", "closeness"]]
        self.centrality_cache_dir = outpath
        if all([os.path.exists(i) for i in outpath]):
            log.debug("中心性存在缓存，直接读取")
            log.info("点度中心性 degree")
            with open(outpath[0], ) as f:
                dc = json.load(f)
            # 中介中心性 betweenness
            log.info("中介中心性 betweenness")
            with open(outpath[1], ) as f:
                bc = json.load(f)
            # 接近中心性 closeness
            log.info("接近中心性 closeness")
            with open(outpath[2], ) as f:
                cc = json.load(f)

        else:
            log.debug("中心性无缓存，直接计算")
            log.info("点度中心性 degree")
            dc = degree_centrality(self.G)
            with open(outpath[0], "w") as f:
                f.write(json.dumps(dc))
            # 中介中心性 betweenness
            log.info("中介中心性 betweenness")
            bc = betweenness_centrality(self.G, weight="weight")
            with open(outpath[1], "w") as f:
                f.write(json.dumps(bc))
            # 接近中心性 closeness
            log.info("接近中心性 closeness")
            cc = closeness_centrality(self.G, distance="weight")
            with open(outpath[2], "w") as f:
                f.write(json.dumps(cc))
        return dc, bc, cc

    def analysis_aa(self, aa: str):
        aa = self._aa2position(aa)
        res = {
            "degree_centrality": self.degree_c[aa],
            "betweenness_centrality": self.betweenness_c[aa],
            "closeness_centrality": self.betweenness_c[aa]
        }
        return res

    def analysis_group(self, group: list):
        # 单点
        # 箱线图-三种中心性、保守性
        aas = [self._aa2position(i) for i in group]

    def _plot_degree_distribuition(self, aas):
        # 度分布 degree
        dh = nx.degree_histogram(self.G)
        log.debug("dh = %s", dh)
        x = list(range(len(dh)))
        y = np.array(dh) / sum(dh)
        fig: plt.Figure
        axs: List[plt.Axes]
        fig, axs = plt.subplots(2, 1)
        # sps1, sps2 = GridSpec(2, 1)
        # fig = plt.figure()
        # hspace 双斜线间隔
        # bax = brokenaxes(ylims=((0, 0.02), (0.8, 0.9)), hspace=0.2, subplot_spec=sps1, fig=fig, height_ratios=(3, 8))
        # bax_axes_botton:plt.Axes = bax.axs[1]
        # bax_axes_botton.set_yticklabels()
        # t = bax_axes_botton.get_yticks()
        # bax_axes_botton.set_yticks(np.arange(0, 0.01, 10))
        # bax_axes_botton.set_yticklabels(np.arange(0, 0.01, 10))
        # log.debug("bax_axes_top.get_ymajorticklabels= %s", bax_axes_botton.get_ymajorticklabels())

        # bax_axes_top.get_ymajorticklabels
        # bax = brokenaxes(ylims=((0, 0.02), (0.8, 0.9)), hspace=0.2, subplot_spec=sps1, fig=fig, )
        # axs = [bax, plt.subplot(sps2)]
        axs[0].plot(x[1:], y[1:])
        axs[0].set_ylabel("degree distribution")
        # axs[0].plot(x, y)
        axs[1].loglog(x, y)
        axs[1].set_ylabel("degree distribution")
        # 将变异的点标注在度分布图中
        ax_0_right: plt.Axes = axs[0].twinx()  # 双坐标轴
        ax_0_right.set_ylabel("important mutations count")
        ax_1_right: plt.Axes = axs[1].twinx()
        ax_1_right.set_ylabel("important mutations count")
        degree2count = dict(zip(x[1:], y[1:]))
        count_degree_aa = defaultdict(int)
        for aa in aas:
            # 拿到节点，并计算度
            node = self.G.nodes[aa]
            degree = self.G.degree[aa]
            # axs[0].annotate(aa, (degree, degree2count[degree]))
            # axs[0].bar(degree, degree2count[degree])
            if degree == 0:
                continue
            count_degree_aa[degree] += 1
        # for degree, count in count_degree_aa.items():
        ax_0_right.bar(count_degree_aa.keys(), count_degree_aa.values(), color="green")
        ax_1_right.bar(count_degree_aa.keys(), count_degree_aa.values(), color="green")

        fig.show()
        fig.savefig(os.path.join(self.data_dir, f"度分布.png"), dpi=300)

    def _plot_node_box(self, aas):
        nodes_size = {node: self.G.nodes[node]["size"] for node in self.G.nodes}
        node_data = pd.DataFrame(
            {"conservation": nodes_size, "degree centrality": self.degree_c, "closeness centrality": self.closeness_c,
             "betweenness centrality": self.betweenness_c})
        # node_data = node_data.reset_index()  # 调整索引
        node_data["is_mutation"] = node_data.index.map(lambda x: x in aas)

        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(4, 1, figsize=(10, 15))
        sns.boxplot(x=node_data["conservation"], y=node_data["is_mutation"], orient="h", ax=axes[0], )
        sns.boxplot(x=node_data["degree centrality"], y=node_data["is_mutation"], orient="h", ax=axes[1], )
        sns.boxplot(x=node_data["closeness centrality"], y=node_data["is_mutation"], orient="h", ax=axes[2])
        sns.boxplot(x=node_data["betweenness centrality"], y=node_data["is_mutation"], orient="h", ax=axes[3])
        plt.show()
        fig.savefig(os.path.join(self.data_dir, "boxplot.png"), dpi=500)
        log.debug("node_data = %s", node_data)

    def _plot_edge_box(self, aas, groups):
        aas2edge = defaultdict(list)
        for group in groups:
            # aas2edge += list(permutations(group, 2))
            for n1, n2 in combinations(group, 2):
                n1 = self._aa2position(n1)
                n2 = self._aa2position(n2)
                aas2edge[n1].append(n2)
                aas2edge[n2].append(n1)
        for k, v in aas2edge.items():
            aas2edge[k] = set(v)

        log.debug("aas2edge = %s", aas2edge)
        edge_info = [[edge[0], edge[1], self.G.edges[edge]["weight"]] for edge in self.G.edges]
        edge_data = pd.DataFrame(edge_info, columns=["node1", "node2", "info"])

        def aux(s):
            # return (s.node1 in aas) and (s.node2 in aas)
            if (s.node1 in aas2edge.keys()) and (s.node2 in aas2edge[s.node1]):
                return True
            elif (s.node2 in aas2edge.keys()) and (s.node1 in aas2edge[s.node2]):
                return True
            else:
                return False

        edge_data["is_mutation"] = edge_data.apply(aux, axis=1)
        log.debug("edge_data[edge_data.is_mutation] = %s", edge_data[edge_data.is_mutation])
        log.debug("edge_data = %s", edge_data)
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, )
        sns.boxplot(x=edge_data["info"], y=edge_data["is_mutation"], orient="h", ax=ax)
        ax.set_xlabel("co-conservation")
        fig.show()
        fig.savefig(os.path.join(self.data_dir, "edge_boxplot.png"), dpi=500)

    def _plot_edge_distribution(self, aas):
        pass

    def analysisG(self, aas: list, groups):
        aas = [self._aa2position(aa) for aa in aas]
        self._plot_degree_distribuition(aas)  # 度分布
        self._plot_node_box(aas, )  # 箱线图：中心性 + 保守性
        self._plot_edge_box(aas, groups)  # 共保守性


if __name__ == '__main__':
    start_time = time.time()
    # 保守性网络
    pcn = ProConNetwork()
    # 需要关注的变异
    groups = AnalysisMutationGroup()
    aas = groups.get_non_duplicated_aas()
    log.debug("aas = %s", aas)

    pcn.analysisG(aas, groups.get_aa_groups())

    end_time = time.time()
    log.info(f"程序运行时间: {end_time - start_time}")
