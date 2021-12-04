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
    def __init__(self, analysis="../data/procon/analysis.json", seed=0, fasta_file="../data/YP_009724390.1.txt", ):
        # 关注的变异组的相关数据
        with open(analysis) as f:
            self.analysis: dict = json.load(f)

        self.seed = seed
        # 氨基酸序列以及所有位点
        self.fasta = next(SeqIO.parse(fasta_file, "fasta")).seq
        self.positions = [f"{i + 1}{aa.upper()}" for i, aa in enumerate(self.fasta)]  # 所有可能的位点
        # 变异组以及对应位点
        self.aa_groups = self.get_aa_groups()
        self.aa_groups_position = self.get_aa_groups_position()
        self.aa_groups_sample = self.resample_groups(self.aa_groups)

        # 不重复变异以及对应位点
        self.non_duplicated_aas = self.get_non_duplicated_aas()
        self.non_duplicated_aas_positions = self.get_non_duplicated_aas_position()
        self.non_duplicated_aas_sample = self.resample_aas(self.non_duplicated_aas_positions)

    @staticmethod
    def aa2position(aa: str):
        if aa[0] in AA and aa[-1] in aa and aa[1:-1].isdigit():
            return aa[1:-1] + aa[0]
        elif aa[:-1].isdigit() and aa[-1] in aa:
            return aa
        else:
            raise RuntimeError(f"无法转化aa={aa}")

    def get_aa_groups(self):
        aas = []
        for i, row in self.analysis.items():
            aas.append(row["aas"])
        return aas

    def get_aa_groups_position(self):
        groups = self.get_aa_groups()
        groups = [[self.aa2position(aa) for aa in group] for group in groups]
        return groups

    def get_non_duplicated_aas(self):
        aas = []
        for i, row in self.analysis.items():
            aas += row["aas"]
        aas = list(set(aas))
        return aas

    def get_non_duplicated_aas_position(self):
        aas = self.get_non_duplicated_aas()
        positions = [self.aa2position(aa) for aa in aas]
        positions = list(set(positions))
        return positions

    def resample_aas(self, aas, N=1000):
        # 重采样
        positions = pd.Series(self.positions)
        log.debug("开始采样，采样数目为%s", N)
        sample_aas = [positions.sample(n=len(aas), random_state=self.seed + i).tolist() for i in range(N)]
        return sample_aas

    def resample_groups(self, groups, N=1000):
        groups = self.aa_groups

        # 重采样
        sample_groups = []
        positions = pd.Series(self.positions)
        log.debug("开始采样，采样数目为%s", N)
        for i, group in enumerate(groups):
            # 采样
            one_sample = [positions.sample(n=len(group), random_state=self.seed + j + i * 1000).tolist() for j in
                          range(N)]
            sample_groups.append(one_sample)
            log.info(f"采样完成{i}, {group}")
        return sample_groups


class ProConNetwork:
    def __init__(self,
                 analysis_mutation_groups: AnalysisMutationGroup,
                 data_dir="../data/procon",
                 parse1="../data/procon/type1_parse.csv",
                 parse2="../data/procon/type2_parse.csv",
                 threshold=100,  # 共保守性的阈值
                 ):
        self.analysis_mutation_group = analysis_mutation_groups

        log.info("构造图...")
        # procon 计算的所有的结果
        self.type1 = pd.read_csv(parse1)  # 单点
        self.type2 = pd.read_csv(parse2)  # 成对
        # 归一化保守性分数
        self.type1_mms, self.type1["info_norm"] = self._normalize_info(self.type1["information"].values)
        self.type2_mms, self.type2["info_norm"] = self._normalize_info(self.type2["info"].values)
        # 网络阈值
        if threshold > 1:
            self.threshold = threshold
        else:
            # 按比例划取阈值
            count_type2 = int(threshold * len(self.type2))
            threshold_score = self.type2["info"].sort_values(ascending=False)[count_type2]
            self.threshold = threshold_score
        log.debug("self.threshold = %s", self.threshold)

        # 输出文件路径
        self.data_dir = os.path.join(data_dir, f"threshold_{threshold}")
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        log.debug("self.data_dir = %s", self.data_dir)

        # fasta 序列，以及位点
        self.fasta = self.analysis_mutation_group.fasta
        self.positions = self.analysis_mutation_group.positions

        # 构建网络
        nodes = self._get_nodes(self.type1)  # 节点
        links = self._get_links(self.type2[self.type2["info"] >= self.threshold])  # 边
        self.G = self._get_G(links, nodes)  # 构成图
        # 添加相邻位点的最低共保守性
        self.G = self._add_neighbour_links(self.G)

        log.debug(f"不同数据的数目:\n\ttype1 {len(self.type1)}\n\tnode {self.G.number_of_nodes()}\n\t"
                  f"type2 {len(self.type2)}\n\tedge {self.G.number_of_edges()}\n\t"
                  f"neighbour edge {self.G.number_of_edges() - len(links)}")

        # 计算中心性
        log.info("确定中心性...")
        self.degree_c, self.betweenness_c, self.closeness_c, self.edge_betweenness_c = self._get_centralities()

    @staticmethod
    def _normalize_info(info: np.ndarray):
        """
        归一化数据到 0 和 1 之间
        :return: 归一器 和 当前数据归一结果
        """
        info = info.reshape(-1, 1)
        mms = preprocessing.MinMaxScaler().fit(info)
        return mms, mms.transform(info)

    @staticmethod
    def _aa2position(aa: str):
        if aa[0] in AA and aa[-1] in aa and aa[1:-1].isdigit():
            return aa[1:-1] + aa[0]
        elif aa[:-1].isdigit() and aa[-1] in aa:
            return aa
        else:
            raise RuntimeError(f"无法转化aa={aa}")

    def _get_nodes(self, type1):
        nodes = self.positions
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

    def _add_neighbour_links(self, G):
        neighbour_links = []
        for i in range(len(self.positions) - 1):
            node1 = self.positions[i]
            node2 = self.positions[i + 1]
            if not G.has_edge(node1, node2):
                neighbour_links.append([node1, node2, self.threshold])

        neighbour_links = pd.DataFrame(neighbour_links, columns=["source", "target", "weight"])
        neighbour_links["weight"] = self.type2_mms.transform(neighbour_links["weight"].values.reshape(-1, 1), )
        log.debug("neighbour_links = %s", neighbour_links)

        # 添加节点
        for i, row in neighbour_links.iterrows():
            G.add_edge(row["source"], row["target"], weight=row["weight"])
        return G

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
        for path in outpath:
            if not os.path.exists(os.path.dirname(path)):
                os.mkdir(os.path.dirname(path))
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

    def apply_to_aas(self, aas, aas_sample, func):
        aas_score = func(aas)
        aas_sample_score = [func(a_sample) for a_sample in aas_sample]
        return aas_score, aas_sample_score

    def _plot_degree_distribuition(self, ):
        """
        度分布 degree
        使用边的权重的平均值作为度
        """
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        aas_sample = self.analysis_mutation_group.non_duplicated_aas_sample

        def calculate_avg_weighted_degree(aas):
            """计算一组位点的平均度"""
            result = []
            for aa in aas:
                # 计算单个位点的加权平均度
                weighted_degrees = []
                for nbr, datadict in self.G.adj[aa].items():
                    weighted_degrees.append(datadict.get("weight", 0))

                if weighted_degrees:
                    avg_weighted_degree = np.mean(weighted_degrees)
                else:
                    avg_weighted_degree = 0
                result.append(avg_weighted_degree)
            return result

        avg_weighted_degrees_result = self.apply_to_aas(aas, aas_sample, calculate_avg_weighted_degree)
        # log.debug("avg_weighted_degrees_result[0] = %s", avg_weighted_degrees_result[0])
        # log.debug("avg_weighted_degrees_result[0][1] = %s", avg_weighted_degrees_result[1][1])
        aas_degrees = pd.Series(avg_weighted_degrees_result[0])
        sample_degrees = pd.Series(np.array(avg_weighted_degrees_result[1]).reshape(-1))
        log.debug("aas_degrees.shape = %s", aas_degrees.shape)
        log.debug("sample_degrees.shape = %s", sample_degrees.shape)

        cut_split = np.arange(0.2, 0.8, 0.1).tolist()
        aas_degrees_cut = pd.cut(aas_degrees, cut_split).value_counts().sort_index() / len(aas_degrees)
        sample_degrees_cut = pd.cut(sample_degrees, cut_split).value_counts().sort_index() / len(sample_degrees)
        log.debug("aas_degrees_cut = %s", aas_degrees_cut)
        log.debug("sample_degrees_cut = %s", sample_degrees_cut)
        # 绘图
        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
        aas_degrees_cut.plot.bar(ax=axes[0])
        axes[0].set_title("degree distribution of mutations")
        axes[0].set_xticklabels(aas_degrees_cut.index, rotation=0)

        sample_degrees_cut.plot.bar(ax=axes[1])
        axes[1].set_title("degree distribution")
        axes[1].set_xticklabels(sample_degrees_cut.index, rotation=0)

        fig.show()
        fig.savefig(os.path.join(self.data_dir, f"度分布.png"), dpi=300)
        # TODO: 绘制箱线图

    def _plot_node_box(self, ):
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        aas_sample = self.analysis_mutation_group.non_duplicated_aas_sample
        aas_sample = np.array(aas_sample).reshape(-1).tolist()  # 扁平化
        plot_data = pd.DataFrame(
            {"position": aas + aas_sample, "is_mutation": [True] * len(aas) + [False] * len(aas_sample)})

        nodes_size = {node: self.G.nodes[node]["size"] for node in self.G.nodes}
        node_data = pd.DataFrame(
            {"conservation": nodes_size, "degree centrality": self.degree_c, "closeness centrality": self.closeness_c,
             "betweenness centrality": self.betweenness_c})

        # node_data = node_data[]
        plot_data["conservation"] = node_data.loc[plot_data["position"], "conservation"].values
        plot_data["degree centrality"] = node_data.loc[plot_data["position"], "degree centrality"].values
        plot_data["closeness centrality"] = node_data.loc[plot_data["position"], "closeness centrality"].values
        plot_data["betweenness centrality"] = node_data.loc[plot_data["position"], "betweenness centrality"].values
        log.debug("plot_data = %s", plot_data)

        def cal_p_mannwhitneyu(data: pd.Series, bo: pd.Series):
            x = data[bo].to_list()
            y = data[bo.apply(lambda x: not x)].to_list()
            p = mannwhitneyu(x, y).pvalue
            return p

        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(4, 1, figsize=(10, 15))
        sns.boxplot(x=plot_data["conservation"], y=plot_data["is_mutation"], orient="h", ax=axes[0], )
        p = cal_p_mannwhitneyu(plot_data["conservation"], plot_data["is_mutation"])
        axes[0].set_xlabel(f"conservtion (p = {p: .3f})")
        sns.boxplot(x=plot_data["degree centrality"], y=plot_data["is_mutation"], orient="h", ax=axes[1], )
        p = cal_p_mannwhitneyu(plot_data["degree centrality"], plot_data["is_mutation"])
        axes[1].set_xlabel(f"degree centrality (p = {p: .3f})")
        sns.boxplot(x=plot_data["closeness centrality"], y=plot_data["is_mutation"], orient="h", ax=axes[2])
        p = cal_p_mannwhitneyu(plot_data["closeness centrality"], plot_data["is_mutation"])
        axes[2].set_xlabel(f"closeness centrality (p = {p: .3f})")
        sns.boxplot(x=plot_data["betweenness centrality"], y=plot_data["is_mutation"], orient="h", ax=axes[3])
        p = cal_p_mannwhitneyu(plot_data["betweenness centrality"], plot_data["is_mutation"])
        axes[3].set_xlabel(f"betweenness centrality (p = {p: .3f})")
        plt.show()
        fig.savefig(os.path.join(self.data_dir, "boxplot.png"), dpi=500)

    def _plot_edge_box(self, ):
        """计算边的相关参数
        分为三类:
        1. 在同一组的 same mutation group
        2. 在所有可能变异的 mutation
        3. 其他 other
        4. 所有 all

        """
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        groups = self.analysis_mutation_group.aa_groups_position
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
        """绘制保守性分数的分布图"""
        log.debug("self.type1.min() = %s", self.type1["information"].min())
        log.debug("self.type1.max() = %s", self.type1["information"].max())
        log.debug("self.type2.min() = %s", self.type2["info"].min())
        log.debug("self.type2.max() = %s", self.type2["info"].max())
        type1_info = self.type1["info_norm"]
        type2_info = self.type2["info_norm"][self.type2["info"] > self.threshold]

        cut_list = np.arange(0, 1.1, 0.1).tolist()
        log.debug("cut_list = %s", cut_list)
        type1_info_count = pd.cut(type1_info, cut_list, ).value_counts().sort_index() / len(type1_info)
        type2_info_count = pd.cut(type2_info, cut_list, ).value_counts().sort_index() / len(type2_info)
        plot_data = pd.DataFrame([
            type1_info_count.to_list() + type2_info_count.to_list(),
            ["conservation"] * len(type1_info_count) + ["co-conservation"] * len(type2_info_count),
            type1_info_count.index.to_list() + type2_info_count.index.to_list(),
            # ])
        ], index=["proportion", "type", "score"]).T
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

    def _collect_mutation_info(self, ):
        """收集变异节点信息"""
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
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

    def calculate_average_shortest_path_length(self, ):
        """平均最短路径长度"""
        # data -> aa : [变异，非变异]
        data = {}
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        aas_sample = self.analysis_mutation_group.non_duplicated_aas_sample
        aas_scores = []
        sample_scores = []
        for aa in aas:
            # 构造数据
            path_len = shortest_path_length(self.G, source=aa)
            path_len = pd.Series(path_len)
            # 在变异中
            aas_scores.append(path_len[aas].mean())
            # 在采样中
            sample_scores.append([path_len[a_sample].mean() for a_sample in aas_sample])

        plot_data = pd.DataFrame(sample_scores, index=aas, )
        plot_data = plot_data.stack().reset_index()
        plot_data.columns = ["aa", "group", "length"]
        fig: plt.Figure
        # axes: List[plt.Axes]
        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # 查看并保存图片
        order_index = np.argsort(aas_scores)
        sns.boxplot(data=plot_data, x="aa", y="length", ax=ax, order=np.array(aas)[order_index])
        ax.scatter(x=range(len(aas)), y=np.array(aas_scores)[order_index])
        ax.set_ylabel("average shortest path length")
        ax.set_xlabel("")
        [txt.set_rotation(90) for txt in ax.get_xticklabels()]

        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(self.data_dir, "average shortest length.png"), dpi=300)

    def analysisG(self, ):
        """绘制相关图表"""
        # self._plot_origin_distribution()  # 绘制保守性的分布情况
        self._collect_mutation_info()  # 收集变异位点的消息，生成表格
        self._plot_procon_distribution()  # 分数分布图
        self._plot_degree_distribuition()  # 度分布
        self._plot_node_box()  # 箱线图：中心性 + 保守性
        self._plot_edge_box()  # 共保守性  TODO: 使用采样的方式
        self.calculate_average_shortest_path_length()

    def random_sample_analysis(self, aas: list, groups, N=1000):
        """使用随机采样的形式，分析实验组和对照组的区别
        :param N: 采样次数
        """
        aas = [self._aa2position(aa) for aa in aas if aa]
        aas = list(set(aas))
        groups = [[self._aa2position(aa) for aa in group] for group in groups]
        log.debug("aas = %s", aas)

        # 重采样
        group_and_sample_groups = []
        positions = pd.Series(self.positions)
        log.info("开始采样，采样数目为%s", N)
        for group in groups:
            # 采样
            sample_groups = [positions.sample(n=len(group), random_state=i).tolist() for i in range(N)]
            group_and_sample_groups.append([group, sample_groups])
            log.info("采样完成")

        def apply_to_group(data, func):
            scores = []
            for group, sample_groups in data:
                s1 = func(group)
                s2s = [func(g) for g in sample_groups]
                scores.append([s1, s2s])
            return scores

        # 度
        # weighted_degrees = self.get_weighted_degree()
        # avg_weighted_degrees = self.get_avg_weighted_degree()

        def calculate_degree(group: List[str]):
            """计算平均度"""
            data = []
            for aa in group:
                degree = self.G.degree[aa]
                data.append(degree)

            # 返回
            return np.mean(data)

        degree_scores = apply_to_group(group_and_sample_groups, calculate_degree)
        # log.debug("degree_scores = %s", degree_scores)
        # log.debug("degree_scores[0][1] = %s", degree_scores[0][1])
        # for i in degree_scores:
        #     print(type(i[1]))

        _plot_scores = []
        for i in degree_scores:
            _plot_scores += i[1]
        _plot_aas = []
        for i in range(len(group_and_sample_groups)):
            _plot_aas += [i, ] * N
        log.debug("len(_plot_aas) = %s", len(_plot_aas))
        log.debug("len(_plot_scores) = %s", len(_plot_scores))
        log.debug("len(degree_scores) = %s", len(degree_scores))
        log.debug("group_and_sample_groups = %s", len(group_and_sample_groups))
        assert len(_plot_aas) == len(_plot_scores)
        # 绘图的数据
        plot_data = pd.DataFrame(
            np.array([_plot_scores, _plot_aas]).T,
            columns=["degree", "position"]
        )
        log.debug("plot_data = %s", plot_data)
        ax: plt.Axes
        fig, ax = plt.subplots()
        sns.boxplot(data=plot_data, x="position", y="degree", ax=ax)
        _plot_normal_degree = [i[0] for i in degree_scores]
        ax.scatter(x=list(range(len(group_and_sample_groups))), y=_plot_normal_degree)
        [txt.set_rotation(90) for txt in ax.get_xticklabels()]
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(self.data_dir, "resample degree.png"), dpi=300)

    def _plot_origin_distribution(self):
        type2 = self.type2["info"]
        type2_count = pd.cut(type2, np.arange(max(type2))).value_counts().sort_index()
        type2_count.index = np.arange(max(type2))[:len(type2_count)]

        # plt.plot(x=type2_count.index, y=type2_count.values)
        ax: plt.Axes
        fig, ax = plt.subplots(figsize=(10, 5))
        type2_count.plot(ax=ax)
        # ax.set_xtick(np.arange(max(type2)))
        fig.tight_layout()
        plt.show()
        fig.savefig(os.path.join(self.data_dir, "共保守性分布情况.png"), dpi=300)


if __name__ == '__main__':
    start_time = time.time()
    # 保守性网络
    # 需要关注的变异
    mutation_groups = AnalysisMutationGroup()
    thresholds = [50, 100, 150, 200, 250, 300]
    for t in thresholds:
        pcn = ProConNetwork(mutation_groups, threshold=t)

        pcn.analysisG()
        # pcn.random_sample_analysis(aas, groups.get_aa_groups())

        end_time = time.time()
        log.info(f"程序运行时间: {end_time - start_time}")
