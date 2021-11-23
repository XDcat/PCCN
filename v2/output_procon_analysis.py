import json
import math
import seaborn as sns
import time
from Bio import SeqIO
import networkx as nx
from networkx.algorithms.centrality import degree_centrality, betweenness_centrality, closeness_centrality, \
    edge_betweenness_centrality
from networkx.algorithms.shortest_paths import shortest_path_length
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from scipy.special import comb
from scipy.stats import mannwhitneyu
from itertools import combinations, permutations
from collections import defaultdict
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
from sklearn import preprocessing
# 日志
import logging
import logconfig
import pickle
from typing import List

logconfig.setup_logging()
log = logging.getLogger("cov2")

AA = ['-', 'C', 'F', 'I', 'L', 'M', 'V', 'W', 'Y', 'A', 'T', 'D', 'E', 'G', 'P', 'N', 'Q', 'S', 'H', 'R', 'K']

sns.set()


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
        log.debug(self.G.edges["734T", "937S"])
        log.debug(self.G.edges["937S", "734T"])
        # 中心性
        log.info("确定中心性...")
        self.degree_c, self.betweenness_c, self.closeness_c, self.edge_betweenness_c = self._get_centralities()

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

    def _get_centralities(self, is_weight=True):
        file_names = ["degree", "betweenness", "closeness", "e_betweenness"]
        # 是否使用带全边
        if is_weight:
            weight = "weight"
            outpath = [os.path.join(self.data_dir, "cache", f"{i}_centrality.json") for i in
                       file_names]
        else:
            weight = None
            outpath = [os.path.join(self.data_dir, "cache", f"{i}_centrality_no_weight.json") for i in
                       file_names]
        # 点度中心性 degree
        self.centrality_cache_dir = outpath

        if not os.path.exists(outpath[0]):
            log.debug("无缓存，直接计算")
            log.info("点度中心性 degree")
            dc = degree_centrality(self.G)
            with open(outpath[0], "w") as f:
                f.write(json.dumps(dc))
        else:
            log.debug("存在缓存，直接读取")
            log.info("点度中心性 degree")
            with open(outpath[0], ) as f:
                dc = json.load(f)

        if not os.path.exists(outpath[1]):
            log.debug("无缓存，直接计算")
            # 中介中心性 betweenness
            log.info("中介中心性 betweenness")
            bc = betweenness_centrality(self.G, weight=weight)
            with open(outpath[1], "w") as f:
                f.write(json.dumps(bc))
        else:
            log.debug("存在缓存，直接读取")
            # 中介中心性 betweenness
            log.info("中介中心性 betweenness")
            with open(outpath[1], ) as f:
                bc = json.load(f)

        if not os.path.exists(outpath[2]):
            log.debug("无缓存，直接计算")
            # 接近中心性 closeness
            log.info("接近中心性 closeness")
            cc = closeness_centrality(self.G, distance=weight)
            with open(outpath[2], "w") as f:
                f.write(json.dumps(cc))
        else:
            log.debug("存在缓存，直接读取")
            # 接近中心性 closeness
            log.info("接近中心性 closeness")
            with open(outpath[2], ) as f:
                cc = json.load(f)

        if not os.path.exists(outpath[3]):
            log.debug("无缓存，直接计算")
            # 边 betweenness
            log.info("边 中介中心性 betweenness")
            e_bc = edge_betweenness_centrality(self.G, weight=weight)
            with open(outpath[3], "wb") as f:
                pickle.dump(e_bc, f)
        else:
            log.debug("存在缓存，直接读取")
            # 边 betweenness
            log.info("边 中介中心性 betweenness")
            with open(outpath[3], "rb") as f:
                e_bc = pickle.load(f)

        return dc, bc, cc, e_bc

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

    def get_degree(self):
        degrees = self.G
        return degrees

    def get_weighted_degree(self):
        """获得加权度"""
        degrees = {}
        # 遍历节点
        for n, nbrs in self.G.adj.items():
            # n 节点，nbrs 邻居节点
            wts = []
            for nbr, eattr in nbrs.items():
                # nbr 邻居节点，eatter 边属性
                wt = eattr["weight"]
                wts.append(wt)
            if wts:
                avg_wt = np.sum(wts)
            else:
                avg_wt = 0.0
            # degrees.append(avg_wt)
            degrees[n] = avg_wt
        return degrees

    def get_avg_weighted_degree(self):
        """获得平均加权度"""
        degrees = {}
        # 遍历节点
        for n, nbrs in self.G.adj.items():
            # n 节点，nbrs 邻居节点
            wts = []
            for nbr, eattr in nbrs.items():
                # nbr 邻居节点，eatter 边属性
                wt = eattr["weight"]
                wts.append(wt)
            if wts:
                avg_wt = np.mean(wts)
            else:
                avg_wt = 0.0
            # degrees.append(avg_wt)
            degrees[n] = avg_wt
        return degrees

    def _plot_degree_distribuition(self, aas):
        # 度分布 degree
        # 使用边的权重的平均值作为度
        degrees = {}
        # 遍历节点
        for n, nbrs in self.G.adj.items():
            # n 节点，nbrs 邻居节点
            wts = []
            for nbr, eattr in nbrs.items():
                # nbr 邻居节点，eatter 边属性
                wt = eattr["weight"]
                wts.append(wt)
            if wts:
                avg_wt = np.mean(wts)
            else:
                avg_wt = 0.0
            # degrees.append(avg_wt)
            degrees[n] = avg_wt
        degrees: pd.Series = pd.Series(degrees)
        log.debug("degrees = %s", degrees)
        log.debug("max(degrees) = %s", max(degrees))
        log.debug("min(degrees) = %s", min(degrees))
        log.debug("np.mean(degrees) = %s", np.mean(degrees))
        # log.debug("pd.qcut(degrees) = %s", pd.cut(degrees, np.arange(0, 0.7, 0.01), right=False))
        cut_split = np.arange(0, 0.7, 0.1)
        degrees_distribution: pd.Series = pd.cut(degrees.values, cut_split, right=False)
        degrees_distribution = degrees_distribution.value_counts()
        degrees_distribution = degrees_distribution / len(degrees)
        log.debug("sum(degrees_distribution) = %s", sum(degrees_distribution))
        # 绘图
        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
        degrees_distribution.plot.bar(ax=axes[0])
        axes[0].set_title("degree distribution")
        axes[0].set_xticklabels(degrees_distribution.index, rotation=0)

        # 变异的分布
        aas_degrees = degrees[aas].sort_index()
        aas_degrees.to_csv(os.path.join(self.data_dir, "aas_degree_distribution.csv"))
        aas_degrees_distribution = pd.cut(aas_degrees.values, cut_split, right=False).value_counts() / len(degrees)
        log.debug("aas_degrees_distribution = %s", aas_degrees_distribution)
        aas_degrees_distribution.plot.bar(ax=axes[1])
        axes[1].set_title("degree distribution of mutations")
        axes[1].set_xticklabels(aas_degrees_distribution.index, rotation=0)
        # log.debug("aas_degrees.sort_values() = %s", aas_degrees.sort_values())

        fig.show()
        fig.savefig(os.path.join(self.data_dir, f"度分布.png"), dpi=300)

    def _plot_node_box(self, aas):
        nodes_size = {node: self.G.nodes[node]["size"] for node in self.G.nodes}
        node_data = pd.DataFrame(
            {"conservation": nodes_size, "degree centrality": self.degree_c, "closeness centrality": self.closeness_c,
             "betweenness centrality": self.betweenness_c})
        node_data["is_mutation"] = node_data.index.map(lambda x: x in aas)
        node_data = node_data[node_data["degree centrality"] > 0]  # 除去孤立节点

        def cal_p_mannwhitneyu(data: pd.Series, bo: pd.Series):
            x = data[bo].to_list()
            y = data[bo.apply(lambda x: not x)].to_list()
            p = mannwhitneyu(x, y).pvalue
            return p

        # node_data = node_data[node_data["conservation"] > 0.3]
        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(4, 1, figsize=(10, 15))
        sns.boxplot(x=node_data["conservation"], y=node_data["is_mutation"], orient="h", ax=axes[0], )
        p = cal_p_mannwhitneyu(node_data["conservation"], node_data["is_mutation"])
        axes[0].set_xlabel(f"conservtion (p = {p: .3f})")
        sns.boxplot(x=node_data["degree centrality"], y=node_data["is_mutation"], orient="h", ax=axes[1], )
        p = cal_p_mannwhitneyu(node_data["degree centrality"], node_data["is_mutation"])
        axes[1].set_xlabel(f"degree centrality (p = {p: .3f})")
        sns.boxplot(x=node_data["closeness centrality"], y=node_data["is_mutation"], orient="h", ax=axes[2])
        p = cal_p_mannwhitneyu(node_data["closeness centrality"], node_data["is_mutation"])
        axes[2].set_xlabel(f"closeness centrality (p = {p: .3f})")
        sns.boxplot(x=node_data["betweenness centrality"], y=node_data["is_mutation"], orient="h", ax=axes[3])
        p = cal_p_mannwhitneyu(node_data["betweenness centrality"], node_data["is_mutation"])
        axes[3].set_xlabel(f"betweenness centrality (p = {p: .3f})")
        plt.show()
        fig.savefig(os.path.join(self.data_dir, "boxplot.png"), dpi=500)
        log.debug("node_data = %s", node_data)

    def _plot_edge_box(self, aas, groups):
        """计算边的相关参数
        分为三类:
        1. 在同一组的 same mutation group
        2. 在所有可能变异的 mutation
        3. 其他 other
        4. 所有 all

        """
        # 可以在不同组的边
        edge_in_different_group = list(permutations(aas, 2))
        # 在同一组的边
        edge_in_same_group = []
        for group in groups:
            group = [self._aa2position(i) for i in group]
            edge_in_same_group += list(permutations(group, 2))
        edge_in_same_group = list(set(edge_in_same_group))  # 去重

        def cal_p_mannwhitneyu(data: pd.Series, x: pd.Series, y: pd.Series):
            x = data.loc[x].to_list()
            y = data.loc[y].to_list()
            p = mannwhitneyu(x, y).pvalue
            return p

        # 初始化 图片
        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))

        # 共保守性
        rows = []
        columns = ["u", "v", "co-conservation", "label"]
        for u, v, weight in self.G.edges.data("weight"):

            # 所有的
            # rows.append([u, v, weight, "all"])

            # 其他三种
            # 先小范围
            if (u, v) in edge_in_same_group:
                rows.append([u, v, weight, "same mutation group"])
            # 再大范围和其他
            if (u, v) in edge_in_different_group:
                rows.append([u, v, weight, "mutation"])
            else:
                rows.append([u, v, weight, "no mutation"])
        co_conservation = pd.DataFrame(rows, columns=columns)
        log.debug("co_conservation.label.value_counts() = %s", co_conservation.label.value_counts())
        sns.boxplot(data=co_conservation, x="co-conservation", y="label", orient="h", ax=axes[0], )
        axes[0].set_xlabel("co-conservation")
        axes[0].set_ylabel("")
        log.debug(
            "p value of co-conservation = %.4f",
            cal_p_mannwhitneyu(
                co_conservation["co-conservation"],
                co_conservation["label"] == "mutation",
                co_conservation["label"] == "no mutation")
        )

        # 边的 betweenness
        rows = []
        columns = ["u", "v", "co-conservation", "label"]
        for (u, v), weight in self.edge_betweenness_c.items():
            # 所有的
            # rows.append([u, v, weight, "all"])
            # 其他三种
            # 先小范围
            if (u, v) in edge_in_same_group:
                rows.append([u, v, weight, "same mutation group"])
            # 再大范围和其他
            if (u, v) in edge_in_different_group:
                rows.append([u, v, weight, "mutation"])
            else:
                rows.append([u, v, weight, "no mutation"])
        co_conservation = pd.DataFrame(rows, columns=columns)
        log.debug("co_conservation.label.value_counts() = %s", co_conservation.label.value_counts())
        sns.boxplot(data=co_conservation, x="co-conservation", y="label", orient="h", ax=axes[1])
        axes[1].set_xlabel("betweenness centrality")
        axes[1].set_ylabel("")

        log.debug(
            "p value of co-conservation = %.4f",
            cal_p_mannwhitneyu(
                co_conservation["co-conservation"],
                co_conservation["label"] == "mutation",
                co_conservation["label"] == "no mutation")
        )
        # 绘图
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(self.data_dir, "edge_boxplot.png"), dpi=500)

    def _plot_procon_distribution(self, ):
        log.debug("self.type1.min() = %s", self.type1["information"].min())
        log.debug("self.type1.max() = %s", self.type1["information"].max())
        log.debug("self.type2.min() = %s", self.type2["info"].min())
        log.debug("self.type2.max() = %s", self.type2["info"].max())
        type1_info = self.type1["info_norm"]
        type2_info = self.type2["info_norm"]

        cut_list = np.arange(0, 1.1, 0.1).tolist()
        log.debug("cut_list = %s", cut_list)
        type1_info_count = pd.cut(type1_info, cut_list, ).value_counts().sort_index() / len(type1_info)
        type2_info_count = pd.cut(type2_info, cut_list, ).value_counts().sort_index() / len(type2_info)
        type1_info_count.plot.bar()
        type2_info_count.plot.bar()
        plot_data = pd.DataFrame([
            type1_info_count.to_list() + type2_info_count.to_list(),
            ["conservation"] * len(type1_info_count) + ["co-conservation"] * len(type2_info_count),
            type1_info_count.index.to_list() + type2_info_count.index.to_list(),
            # ])
        ], index=["proportion", "type", "score"]).T
        # fig: plt.Figure = plt.figure()
        # ax: plt.Axes = fig.add_subplot()
        fig = sns.catplot(
            kind="bar",
            x="type",
            y="proportion",
            hue="score",
            data=plot_data,
            height=10,
        )
        plt.show()
        fig.savefig(os.path.join(self.data_dir, "procon distribution.png"), dpi=500)

    def _collect_mutation_info(self, aas):
        """收集变异节点信息"""
        weighted_degrees = self.get_weighted_degree()
        avg_weighted_degrees = self.get_avg_weighted_degree()

        data = []
        for aa in aas:
            # 度
            degree = self.G.degree[aa]
            # 加权度
            w_degree = weighted_degrees[aa]
            # 加权平均度
            avg_w_degrees = avg_weighted_degrees[aa]
            # conservation
            if aa in self.type1.position.to_list():
                conservation = self.type1["information"][self.type1.position == aa].to_list()[0]
                norm_conservation = self.type1["info_norm"][self.type1.position == aa].to_list()[0]
            else:
                log.debug("%s 不在列表中", aa)
                conservation = 0
                norm_conservation = 0

            # 中心性
            # self.closeness_c[]
            # self.betweenness_c[]
            # self.closeness_c[]

            data.append(
                {"aa": aa, "degree": degree, "weighted degree": w_degree, "average weighted degree": avg_w_degrees,
                 "conservation": conservation, "normalized conservation": norm_conservation,
                 "closeness centrality": self.closeness_c[aa],
                 "betweenness centrality": self.betweenness_c[aa],
                 "degree centrality": self.degree_c[aa],
                 })

        data = pd.DataFrame(data)
        data = data.set_index("aa", drop=False).sort_index()
        data.to_csv(os.path.join(self.data_dir, "aas_info.csv"))


    def calculate_average_shortest_path_length(self, aas):
        """平均最短路径长度"""
        # data -> aa : [变异，非变异]
        data = {}
        for aa in aas:
            mutation_len = []  # 与其他变异相连的路径
            other_len = []  # 与其他相连的路径
            path_len: dict = shortest_path_length(self.G, source=aa)
            for target, length in path_len.items():
                if target in aas:
                    mutation_len.append(length)
                else:
                    other_len.append(length)
            average_mutation_len = 0 if len(mutation_len) == 0 else np.mean(mutation_len)
            average_other_len = 0 if len(other_len) == 0 else np.mean(other_len)
            data[aa] = {
                "mutation": average_mutation_len,
                "other": average_other_len
            }
        data = pd.DataFrame(data).T
        log.debug("data = %s", data.head())
        p_value = mannwhitneyu(data.iloc[:, 0], data.iloc[:, 1]).pvalue
        data.to_csv(os.path.join(self.data_dir, "mutations' average shortest path length.csv"))
        data.index.name = "position"
        plt_data = data.stack().reset_index()
        plt_data.columns = ["position", "type", "length"]
        plt_data = plt_data.sort_values("position")  # 排序
        log.debug("plt_data = %s", plt_data.head())

        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        sns.barplot(data=plt_data, x="position", y="length", hue="type", ax=axes[0], )
        sns.boxplot(data=plt_data, x="type", y="length", ax=axes[1])
        # 调整图像
        [txt.set_rotation(90) for txt in axes[0].get_xticklabels()]  # 字体旋转
        axes[0].legend()
        axes[0].set_xlabel("")
        axes[1].set_xlabel(f"p = {p_value:.3f}")
        # fig.suptitle("average shortest path length", va="bottom")
        fig.tight_layout()
        # 查看并保存图片
        fig.show()
        fig.savefig(os.path.join(self.data_dir, "average shortest length.png"), dpi=300)

    def analysisG(self, aas: list, groups):
        aas = [self._aa2position(aa) for aa in aas if aa]
        aas = list(set(aas))
        log.debug("aas = %s", aas)

        # self._collect_mutation_info(aas)  # 收集变异位点的信息
        # self._plot_procon_distribution()  # 分数分布图
        # self._plot_degree_distribuition(aas)  # 度分布
        # self._plot_node_box(aas, )  # 箱线图：中心性 + 保守性
        # self._plot_edge_box(aas, groups)  # 共保守性
        self.calculate_average_shortest_path_length(aas)


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
