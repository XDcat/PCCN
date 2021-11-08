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
from itertools import combinations

# 日志
import logging
import logconfig

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
                 analysis="../data/procon/analysis.json",
                 parse1="../data/procon/type1_parse.csv",
                 parse2="../data/procon/type2_parse.csv",
                 fasta_file="../data/YP_009724390.1.txt",
                 procon_threshold=300,
                 ):
        self.data_dir = data_dir
        self.procon_threhold = procon_threshold  # 确定网络阈值

        log.info("构造图...")
        # procon 计算的所有的结果
        self.type1 = pd.read_csv(parse1)  # 单点
        self.type2 = pd.read_csv(parse2)  # 成对

        # fasta 序列
        self.fasta = next(SeqIO.parse(fasta_file, "fasta")).seq

        # 节点
        self.nodes = self._get_nodes(self.fasta, self.type1)
        # 边
        self.links = self._get_links(self.type2)
        # 构成图
        self.G = self._get_G(self.links, self.nodes)

        log.info("确定中心性...")
        # 中心性
        self.degree_c, self.betweenness_c, self.closeness_c = self._get_centralities()

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
        nodes = nodes.loc[:, ["position", "information"]]
        nodes.columns = ["name", "size"]
        nodes = nodes.fillna(0)
        log.debug("nodes = %s", nodes)
        return nodes

    def _get_links(self, type2, ):
        type2 = type2[type2["info"] > self.procon_threhold]  # TODO: 选择阈值
        log.debug("type2 = %s", type2)
        links = type2.loc[:, ["site1", "site2", "info"]]
        links.columns = ["source", "target", "weight"]
        log.debug("links = %s", links)
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
        log.info("nodes = %s", self.nodes)
        log.info("links = %s", self.links)

    @staticmethod
    def count_distribution(d: dict):
        values = list(d.values())
        counts = pd.value_counts(values)
        counts = counts.sort_index()
        counts = counts.to_dict()
        return counts.keys(), counts.values()

    def _get_centralities(self):
        # 点度中心性 degree
        dc = degree_centrality(self.G)
        # 中介中心性 betweenness
        bc = betweenness_centrality(self.G)
        # 接近中心性 closeness
        cc = closeness_centrality(self.G)
        # log.debug("dc = %s", dc)
        # log.debug("cc = %s", cc)
        # log.debug("bc = %s", bc)
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
        aas = [self._aa2position(i) for i in group]
        nodes_size = [self.G.nodes[aa]["size"] for aa in aas]
        nodes_mean = np.mean(nodes_size)  # 平均值
        nodes_var = np.var(nodes_size)  # 方差

        edges_size = []
        for aa1, aa2 in combinations(aas, 2):
            if self.G.has_edge(aa1, aa2):
                w = self.G.edges[aa1, aa2]["weight"]
            else:
                w = 0
            edges_size.append(w)
        edges_mean = np.mean(edges_size)
        edges_var = np.var(edges_size)

        res = {
            "nodes_mean": nodes_mean,
            "nodes_var": nodes_var,
            "edges_mean": edges_mean,
            "edges_var": edges_var
        }
        return res

    def analysisG(self):
        # type2 分数的分布情况
        # 获取 type2 的 info
        # type2_info = pcn.type2["info"]
        type2_info = pcn.type2["info"].astype(int)
        type2_info_cut: pd.Series = pd.cut(type2_info, bins=range(
            0,
            (int(max(type2_info)) // 10 + 1) * 10,
            10
        ))
        # 统计数目并排序
        type2_info_distribution = type2_info_cut.value_counts().sort_index()
        log.debug("len(self.type2) = %s", len(self.type2))
        log.debug("type2_info_distribution = %s", type2_info_distribution)
        x = type2_info_distribution.index.astype(str)
        y = type2_info_distribution.values
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle("Distribution of ProCon information")
        ax.plot(x, y)
        plt.xticks(rotation=90)
        fig.show()
        fig.savefig(os.path.join(self.data_dir, "共保守分数分布情况"))

        # 度分布 degree
        dh = nx.degree_histogram(self.G)
        log.debug("dh = %s", dh)
        x = list(range(len(dh)))
        y = np.array(dh) / sum(dh)
        fig, axs = plt.subplots(2, 1)
        fig.suptitle("threshold={}".format(self.procon_threhold))
        axs[0].plot(x[1:], y[1:])
        axs[1].loglog(x, y)
        fig.show()
        fig.savefig(os.path.join(self.data_dir, f"度分布-{self.procon_threhold}.png"))


if __name__ == '__main__':
    start_time = time.time()
    # 保守性网络
    pcn = ProConNetwork(procon_threshold=300)
    pcn.analysisG()
    # 需要关注的变异
    groups = AnalysisMutationGroup()
    aas = groups.get_non_duplicated_aas()
    for aa in aas:
        res = pcn.analysis_aa(aa)
        log.debug("%s %s", aa, res)

    for group in groups.get_aa_groups():
        res = pcn.analysis_group(group)
        log.debug("%s\t%s", group, res)

    end_time = time.time()
    log.info(f"程序运行时间: {end_time - start_time}")
