import json
# 日志
import logging
import math
import os
import pickle
import re
import time
from collections import defaultdict
from functools import reduce
from itertools import combinations, permutations
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from networkx.algorithms.centrality import degree_centrality, betweenness_centrality, closeness_centrality, \
    edge_betweenness_centrality
from networkx.algorithms.shortest_paths import shortest_path_length
from pyecharts import options as opts
from pyecharts.charts import Graph
# 组合数
from scipy.special import comb
from scipy.stats import mannwhitneyu, ttest_1samp
from sklearn import preprocessing
from statannotations.Annotator import Annotator

import logconfig

logconfig.setup_logging()
log = logging.getLogger("cov2")

AA = ['-', 'C', 'F', 'I', 'L', 'M', 'V', 'W', 'Y', 'A', 'T', 'D', 'E', 'G', 'P', 'N', 'Q', 'S', 'H', 'R', 'K']

# sns.set()
sns.set_style("ticks")
plt.rcParams.update({'font.size': 16})
plt.rcParams["axes.titlesize"] = "medium"


# scaler
def scaler(d: dict):
    min_max_scaler = preprocessing.MinMaxScaler()
    keys = d.keys()
    # values = np.array(d.values(), dtype=float)
    values = np.array(list(d.values())).reshape(-1, 1)
    values = min_max_scaler.fit_transform(values)
    values = values.reshape(-1).tolist()
    res = dict(zip(keys, values))
    return res


class AnalysisMutationGroup:
    def __init__(self, analysis="../data/procon/analysis.json", seed=0, fasta_file="../data/YP_009724390.1.txt", ):
        with open(analysis) as f:
            self.analysis: dict = json.load(f)

        self.seed = seed
        self.fasta = next(SeqIO.parse(fasta_file, "fasta")).seq
        self.positions = [f"{i + 1}{aa.upper()}" for i, aa in enumerate(self.fasta)]

        self.non_duplicated_aas = self.get_non_duplicated_aas()
        self.non_duplicated_aas_positions = self.get_non_duplicated_aas_position()
        self.non_duplicated_aas_sample = self.resample_aas(self.non_duplicated_aas_positions)

        self.aa_groups, self.aa_groups_info = self.get_aa_groups()
        self.aa_groups_position = self.get_aa_groups_position()
        self.group_count_sample = self.resample_groups()

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
        names = []
        categroies = []
        for i, row in self.analysis.items():
            aas.append(row["aas"])
            categroies.append(row["WHO label"])
            # categroies.append(row["Category"])
            if type(row["WHO label"]) == str:
                # name = "{} ({})".format(row["Lineage + additional mutations"], row["WHO label"])
                name = "{}({})".format(row["Lineage + additional mutations"], row["WHO label"])
            else:
                name = "{}".format(row["Lineage + additional mutations"])
            names.append(name)

        info = pd.DataFrame({"name": names, "category": categroies})
        # color
        info["color"] = sns.hls_palette(n_colors=len(info), )
        return aas, info

    def get_aa_groups_position(self):
        groups, _ = self.get_aa_groups()
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
        # resample
        positions = list(set(self.positions) - set(aas))
        positions = sorted(positions)
        positions = pd.Series(positions)
        log.debug("count %s", N)
        # sample_aas = [positions.sample(n=len(aas), random_state=self.seed + i).tolist() for i in range(N)]
        sample_aas = [positions.sample(n=len(aas), random_state=self.seed + i).tolist() for i in range(N)]
        return sample_aas

    def resample_groups(self, N=1000):
        """
        resample
        """
        groups = self.aa_groups
        positions = list(set(self.positions) - set(self.non_duplicated_aas_positions))
        positions = sorted(positions)
        positions = pd.Series(positions)

        sample_groups = {}
        group_counts = np.unique([len(group) for group in groups])
        log.debug("count %s", N)
        for count in group_counts:
            one_sample = [positions.sample(n=count, random_state=self.seed + j + count * 1000).tolist() for j in
                          range(N)]
            sample_groups[count] = one_sample
            log.debug(f"group len =  {count}")
        return sample_groups

    def display_seq_and_aa(self):
        aas = self.non_duplicated_aas
        aas = sorted(aas, key=lambda x: int(x[1:-1]))
        log.info(f"fasta({len(self.fasta)}): {self.fasta}")
        count = [len(group) for group in self.aa_groups_position]
        log.debug("pd.value_counts(count).sort_index() = %s", pd.value_counts(count).sort_index())

        all_aa = reduce(lambda x, y: x + y, self.aa_groups_position)
        log.debug("all aa count = %s", pd.value_counts(all_aa))

        log.info("number of variation: %s", len(self.aa_groups))
        log.info("number of aas: %s", len(self.non_duplicated_aas))
        log.info("number of site: %s", len(self.non_duplicated_aas_positions))

    def count_aa(self):
        group = self.aa_groups
        group_info = self.aa_groups_info
        names = group_info["name"].to_list()

        result = {}
        for i in range(len(group)):
            name = names[i]
            N = len(group[i])
            if N not in result:
                result[N] = {"Count": 0, "Variant": []}
            result[N]["Count"] += 1
            result[N]["Variant"].append(name)

        for k in result.keys():
            result[k]["Variant"] = ", ".join(result[k]["Variant"])

        result = pd.DataFrame(result).T
        result = result.sort_index()
        result.index.name = "Number of substitutions"
        result = result.rename(columns={"Count": "Number of variants"})
        result.to_excel(os.path.join("../data/procon", "aa count.xlsx"))


class ProConNetwork:
    def __init__(self,
                 analysis_mutation_groups: AnalysisMutationGroup,
                 data_dir="../data/procon",
                 parse1="../data/procon/type1_parse.csv",
                 parse2="../data/procon/type2_parse.csv",
                 threshold=100,
                 ):
        self.analysis_mutation_group = analysis_mutation_groups

        log.info("构造图...")
        # read data
        self.type1 = pd.read_csv(parse1)
        self.type2 = pd.read_csv(parse2)
        self.type1_mms, self.type1["info_norm"] = self._normalize_info(self.type1["information"].values)
        self.type2_mms, self.type2["info_norm"] = self._normalize_info(self.type2["info"].values)

        if threshold > 1:
            self.threshold = threshold
        else:
            count_type2 = int(threshold * len(self.type2))
            threshold_score = self.type2["info"].sort_values(ascending=False)[count_type2]
            self.threshold = threshold_score
        log.debug("self.threshold = %s", self.threshold)

        # ouput dir
        self.data_dir = os.path.join(data_dir, f"threshold_{threshold}")
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        log.debug("self.data_dir = %s", self.data_dir)

        # fasta position
        self.fasta = self.analysis_mutation_group.fasta
        self.positions = self.analysis_mutation_group.positions

        # construct network
        nodes = self._get_nodes(self.type1)
        links = self._get_links(self.type2[self.type2["info"] >= self.threshold])
        self.G = self._get_G(links, nodes)
        self.G = self._add_neighbour_links(self.G)

        log.debug(f"\n\ttype1 {len(self.type1)}\n\tnode {self.G.number_of_nodes()}\n\t"
                  f"type2 {len(self.type2)}\n\tedge {self.G.number_of_edges()}\n\t"
                  f"neighbour edge {self.G.number_of_edges() - len(links)}")

        # centrality
        self.degree_c, self.betweenness_c, self.closeness_c, self.edge_betweenness_c = self._get_centralities()
        self.degree = self.calculate_degree()

        # page rank
        log.info("page rank ...")
        self.page_rank = nx.pagerank(self.G)
        log.info("shortest path length ...")
        self.shortest_path_length = self.get_shortest_path_length()

    @staticmethod
    def _normalize_info(info: np.ndarray):
        """
        normalize
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
            raise RuntimeError(f"error aa={aa}")

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

        # add node
        for i, row in neighbour_links.iterrows():
            G.add_edge(row["source"], row["target"], weight=row["weight"])
        return G

    def _get_G(self, links, nodes):
        # draw relationship
        G = nx.Graph()
        for i, row in nodes.iterrows():
            if row["name"] in self.analysis_mutation_group.non_duplicated_aas_positions:
                G.add_node(row["name"], size=row["size"], is_mutation=True, pst=int(row["name"][:-1]))
            else:
                G.add_node(row["name"], size=row["size"], is_mutation=False, pst=int(row["name"][:-1]))

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
        self.centrality_cache_dir = outpath

        if not os.path.exists(outpath[0]):
            log.debug("no cache")
            log.info("degree")
            dc = degree_centrality(self.G)
            with open(outpath[0], "w") as f:
                f.write(json.dumps(dc))
        else:
            log.debug("with cache")
            log.info("degree")
            with open(outpath[0], ) as f:
                dc = json.load(f)

        if not os.path.exists(outpath[1]):
            log.debug("no cache")
            log.info("betweenness")
            bc = betweenness_centrality(self.G, weight=weight)
            with open(outpath[1], "w") as f:
                f.write(json.dumps(bc))
        else:
            log.debug("with cache")
            # betweenness
            log.info("betweenness")
            with open(outpath[1], ) as f:
                bc = json.load(f)

        if not os.path.exists(outpath[2]):
            log.debug("no cache")
            # closeness
            log.info("closeness")
            cc = closeness_centrality(self.G, distance=weight)
            with open(outpath[2], "w") as f:
                f.write(json.dumps(cc))
        else:
            log.debug("with cache")
            # closeness
            log.info("closeness")
            with open(outpath[2], ) as f:
                cc = json.load(f)

        if not os.path.exists(outpath[3]):
            log.debug("no cache")
            #  betweenness
            log.info("betweenness")
            e_bc = edge_betweenness_centrality(self.G, weight=weight, normalized=False)
            with open(outpath[3], "wb") as f:
                pickle.dump(e_bc, f)
        else:
            log.debug("with chche")
            # betweenness
            log.info("betweenness")
            with open(outpath[3], "rb") as f:
                e_bc = pickle.load(f)

        result = [dc, bc, cc, e_bc]
        # result = [scaler(i) for i in result]
        result = tuple(result)
        return result

    def analysis_aa(self, aa: str):
        aa = self._aa2position(aa)
        res = {
            "degree_centrality": self.degree_c[aa],
            "betweenness_centrality": self.betweenness_c[aa],
            "closeness_centrality": self.betweenness_c[aa]
        }
        return res

    def analysis_group(self, group: list):
        aas = [self._aa2position(i) for i in group]

    def get_degree(self):
        degrees = self.G
        return degrees

    def get_weighted_degree(self):
        degrees = {}
        for n, nbrs in self.G.adj.items():
            wts = []
            for nbr, eattr in nbrs.items():
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
        degrees = {}
        for n, nbrs in self.G.adj.items():
            wts = []
            for nbr, eattr in nbrs.items():
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
        degree distribution
        """
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        aas_sample = self.analysis_mutation_group.non_duplicated_aas_sample

        def calculate_avg_weighted_degree(aas):
            result = []
            for aa in aas:
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
        log.debug("min(aas_degrees) = %s", min(aas_degrees))
        log.debug("max(aas_degrees) = %s", max(aas_degrees))
        log.debug("min(sample_degrees) = %s", min(sample_degrees))
        log.debug("max(sample_degrees) = %s", max(sample_degrees))

        # find min max
        min_value = min(min(aas_degrees), min(sample_degrees))
        max_value = max(max(aas_degrees), max(sample_degrees))
        min_value = math.floor(min_value * 10) / 10
        max_value = math.ceil(max_value * 10) / 10 + 0.03
        cut_split = np.arange(min_value, max_value, 0.1).tolist()
        log.debug("cut_split = %s", cut_split)

        # aas_degrees.sort_values().reset_index(drop=True).plot()
        # plt.show()
        # sample_degrees.sort_values().reset_index(drop=True).plot()
        # plt.show()
        # plt.loglog(aas_degrees.sort_values().values)
        # plt.show()
        # plt.loglog(sample_degrees.sort_values().values)
        # plt.show()

        # aas_degrees_cut = pd.cut(aas_degrees, cut_split).value_counts().sort_index() / len(aas_degrees)
        # sample_degrees_cut = pd.cut(sample_degrees, cut_split).value_counts().sort_index() / len(sample_degrees)
        # log.debug("aas_degrees_cut = %s", aas_degrees_cut)
        # log.debug("sample_degrees_cut = %s", sample_degrees_cut)
        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.8), sharex=True, sharey=True, constrained_layout=True)
        # aas_degrees_cut.plot.bar(ax=axes[0])
        sns.histplot(aas_degrees, stat="probability", ax=axes[0], bins=cut_split)
        axes[0].set_xlabel("variant")
        # axes[0].set_xticklabels(aas_degrees_cut.index, rotation=0)

        # sample_degrees_cut.plot.bar(ax=axes[1])
        sns.histplot(sample_degrees, stat="probability", ax=axes[1], bins=cut_split)
        axes[1].set_xlabel("sample")
        # axes[1].set_xticklabels(sample_degrees_cut.index, rotation=0)

        fig.show()
        # fig.tight_layout()
        fig.savefig(os.path.join(self.data_dir, f"度分布.png"), dpi=300)

    def _plot_node_centraility_box(self, ):
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        aas_sample = self.analysis_mutation_group.non_duplicated_aas_sample
        aas_sample = np.array(aas_sample).reshape(-1).tolist()  # 扁平化
        plot_data = pd.DataFrame(
            {"position": aas + aas_sample, "is_variant": [True] * len(aas) + [False] * len(aas_sample)})
        plot_data["tag"] = plot_data["is_variant"].apply(lambda x: "variant" if x else "sample")

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
        fig, axes = plt.subplots(1, 3, figsize=(11, 4.8), constrained_layout=True)
        sns.boxplot(y=plot_data["degree centrality"], x=plot_data["tag"], ax=axes[0], fliersize=1)
        p = cal_p_mannwhitneyu(plot_data["degree centrality"], plot_data["is_variant"])
        # axes[0].set_title(f"degree centrality (p = {p: .3f})")
        axes[0].set_title(f"p = {p: .3f}")
        axes[0].set_xlabel("degree centrality")
        axes[0].set_ylim(axes[0].get_ylim()[0], 0.55)

        sns.boxplot(y=plot_data["closeness centrality"], x=plot_data["tag"], ax=axes[1], fliersize=1)
        p = cal_p_mannwhitneyu(plot_data["closeness centrality"], plot_data["is_variant"])
        # axes[1].set_title(f"closeness centrality (p = {p: .3f})")
        axes[1].set_title(f"p = {p: .3f}")
        axes[1].set_xlabel(f"closeness centrality")

        sns.boxplot(y=plot_data["betweenness centrality"], x=plot_data["tag"], ax=axes[2], fliersize=1)
        p = cal_p_mannwhitneyu(plot_data["betweenness centrality"], plot_data["is_variant"])
        axes[2].set_title(f"p = {p: .3f}")
        axes[2].set_xlabel(f"betweenness centrality")
        axes[2].set_ylim(axes[2].get_ylim()[0], 0.0145)

        [ax.set_ylabel("") for ax in axes]
        # [ax.set_xlabel("") for ax in axes]
        plt.show()
        fig.savefig(os.path.join(self.data_dir, "centraility boxplot.png"), dpi=500)

    def _plot_conservation_box(self, ):
        """figures for edge
        3 types
        1.  same mutation group
        2. mutation
        3. other
        4. all
        """
        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(7, 4.8))
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        aas_sample = self.analysis_mutation_group.non_duplicated_aas_sample
        aas_sample = np.array(aas_sample).reshape(-1).tolist()
        plot_data = pd.DataFrame(
            {"position": aas + aas_sample, "is_variant": [True] * len(aas) + [False] * len(aas_sample)})
        plot_data["tag"] = plot_data["is_variant"].apply(lambda x: "variant" if x else "sample")

        nodes_size = {node: self.G.nodes[node]["size"] for node in self.G.nodes}
        node_data = pd.DataFrame(
            {"conservation": nodes_size, "degree centrality": self.degree_c, "closeness centrality": self.closeness_c,
             "betweenness centrality": self.betweenness_c})

        plot_data["conservation"] = node_data.loc[plot_data["position"], "conservation"].values

        def cal_p_mannwhitneyu(data: pd.Series, bo: pd.Series):
            x = data[bo].to_list()
            y = data[bo.apply(lambda x: not x)].to_list()
            p = mannwhitneyu(x, y).pvalue
            return p

        sns.boxplot(y=plot_data["conservation"], x=plot_data["tag"], ax=axes[0], fliersize=1)
        p = cal_p_mannwhitneyu(plot_data["conservation"], plot_data["is_variant"])
        axes[0].set_title(f"p = {p: .3f}")
        axes[0].set_xlabel(f"conservation")
        axes[0].set_ylabel("")

        # CCR
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        groups = self.analysis_mutation_group.aa_groups_position
        edge_in_different_group = list(permutations(aas, 2))
        edge_in_same_group = []
        for group in groups:
            group = [self._aa2position(i) for i in group]
            edge_in_same_group += list(permutations(group, 2))
        edge_in_same_group = list(set(edge_in_same_group))

        def cal_p_mannwhitneyu(data: pd.Series, x: pd.Series, y: pd.Series):
            x = data.loc[x].to_list()
            y = data.loc[y].to_list()
            p = mannwhitneyu(x, y).pvalue
            return p

        # co-conservation
        rows = []
        columns = ["u", "v", "co-conservation", "label"]
        for u, v, weight in self.G.edges.data("weight"):

            # all
            # rows.append([u, v, weight, "all"])

            # other
            if (u, v) in edge_in_same_group:
                rows.append([u, v, weight, "same variant group"])
            if (u, v) in edge_in_different_group:
                rows.append([u, v, weight, "variant"])
            else:
                rows.append([u, v, weight, "sample"])
        co_conservation = pd.DataFrame(rows, columns=columns)
        co_conservation = co_conservation[co_conservation["label"] != "same variant group"]
        log.debug("co_conservation.label.value_counts() = %s", co_conservation.label.value_counts())
        sns.boxplot(data=co_conservation, y="co-conservation", x="label", ax=axes[1], fliersize=1,
                    order=["variant", "sample", ],
                    # hue_order=["sampled nodes", "variant nodes"]
                    )
        p_value = cal_p_mannwhitneyu(
            co_conservation["co-conservation"],
            co_conservation["label"] == "variant nodes",
            co_conservation["label"] == "sampled nodes")
        log.debug("p_value = %s", p_value)
        log.debug("p_value = %s", p_value)
        axes[1].set_title(f"p = {p_value:.3f}")
        axes[1].set_xlabel(f"co-conservation")
        axes[1].set_ylabel("")

        [ax.set_ylabel("") for ax in axes]
        # [ax.set_xlabel("") for ax in axes]
        fig.show()
        fig.savefig(os.path.join(self.data_dir, "edge_boxplot.png"), dpi=500)

    def _plot_procon_distribution(self, ):
        """procon distribution"""
        log.debug("self.type1.min() = %s", self.type1["information"].min())
        log.debug("self.type1.max() = %s", self.type1["information"].max())
        log.debug("self.type2.min() = %s", self.type2["info"].min())
        log.debug("self.type2.max() = %s", self.type2["info"].max())
        type1_info = self.type1["info_norm"]
        type2_info = self.type2["info_norm"]
        data = pd.DataFrame({"score": pd.concat([type1_info, type2_info]).reset_index(drop=True),
                             "kind": ["CR"] * len(type1_info) + ["CCR"] * len(type2_info)})
        axes: List[plt.Axes]
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 4.8))
        sns.histplot(data=data[data["kind"] == "CR"], x="score", stat="probability", bins=20, ax=axes[0], )
        sns.histplot(data=data[data["kind"] == "CCR"], x="score", stat="probability", bins=20, ax=axes[1], )
        # axes[0].set_xlabel("")
        # axes[1].set_xlabel("")
        axes[0].set_xlabel("CS")
        axes[1].set_xlabel("CCS")
        fig.show()
        fig.savefig(os.path.join(self.data_dir, "Figure 1 Distribution of conservation.png"), dpi=500)

    def _collect_mutation_info(self, save=True):
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        funcs: dict = self.get_functions()
        funcs.pop("co-conservation")
        funcs.pop("average shortest length")
        data = {n: f(aas) for n, f in funcs.items()}
        data["aa"] = aas
        data = pd.DataFrame(data)
        data = data.set_index("aa", drop=False).sort_index()
        if save:
            data.to_csv(os.path.join(self.data_dir, "aas_info.csv"))
        return data

    def calculate_average_shortest_path_length(self, ):
        # data -> aa : [mutation，no-mutation]
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        aas_sample = self.analysis_mutation_group.non_duplicated_aas_sample
        aas_scores = []
        sample_scores = []
        for aa in aas:
            path_len = shortest_path_length(self.G, source=aa)
            path_len = pd.Series(path_len)
            aas_scores.append(path_len[aas].mean())
            sample_scores.append([path_len[a_sample].mean() for a_sample in aas_sample])

        plot_data = pd.DataFrame(sample_scores, index=aas, )
        plot_data = plot_data.stack().reset_index()
        plot_data.columns = ["aa", "group", "length"]
        # save detailed
        detail_data = pd.DataFrame(
            {"aa": aas, "aas_avg_shortest_length": aas_scores, "sample_avg_shortest_length": sample_scores,
             "position": [int(i[:-1]) for i in aas]}
        ).sort_values(by="position", ).to_csv(os.path.join(self.data_dir, "averrage shortest length.csv"))

        fig: plt.Figure
        # axes: List[plt.Axes]
        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(14, 4.8))
        # save
        order_index = np.argsort(aas_scores)
        sns.boxplot(data=plot_data, x="aa", y="length", ax=ax, order=np.array(aas)[order_index], fliersize=1)
        ax.scatter(x=range(len(aas)), y=np.array(aas_scores)[order_index])
        # ax.set_ylabel("average shortest path length")
        ax.set_ylabel("")
        ax.set_xlabel("")
        [txt.set_rotation(90) for txt in ax.get_xticklabels()]

        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(self.data_dir, "average shortest length.png"), dpi=300)

        is_bigger = {}
        for i in range(len(aas)):
            aa = aas[i]
            score = aas_scores[i]
            s_scores = sample_scores[i]
            percentile_75 = np.percentile(s_scores, 75)
            is_bigger[aa] = score >= percentile_75
        log.debug("is_bigger = %s", is_bigger)

        # heatmap
        heatmap_data = []
        aa_groups = self.analysis_mutation_group.aa_groups_position
        aa_groups_names = self.analysis_mutation_group.aa_groups_info["name"]
        for group in aa_groups:
            row = []
            for aa, flag in is_bigger.items():
                if aa in group:
                    if flag:
                        row.append(1)
                    else:
                        row.append(0)
                else:
                    row.append(0)
            heatmap_data.append(row)
        heatmap_data = pd.DataFrame(heatmap_data, index=aa_groups_names, columns=aas)
        log.debug("heatmap_data = %s", heatmap_data)

        fig = plt.figure(figsize=(10, 15))
        ax = fig.subplots()
        sns.heatmap(heatmap_data, ax=ax)
        fig.tight_layout()
        fig.show()

        is_bigger_position = set([k for k, v in is_bigger.items() if v])
        is_bigger_ratio = []
        for group in aa_groups:
            is_bigger_in_group = set(group) & is_bigger_position
            is_bigger_ratio.append(
                len(is_bigger_in_group) / len(group)
            )
        log.debug("is_bigger_ratio = %s", is_bigger_ratio)
        barplot_data = pd.DataFrame({"y": is_bigger_ratio, })
        barplot_data = pd.concat([barplot_data, self.analysis_mutation_group.aa_groups_info], axis=1)
        log.debug("barplot_data = %s", barplot_data)

        fig = plt.figure(figsize=(15, 10))
        ax = fig.subplots()
        sns.barplot(data=barplot_data, x="name", y="y", hue="category")
        [txt.set_rotation(90) for txt in ax.get_xticklabels()]
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(self.data_dir, "平均最短路径长度 较大值占据毒株比例.png"), dpi=300)

    def analysisG(self, ):
        # self._plot_origin_distribution()  # procon distribution
        # self._plot_mutations_relationship()  # mutation relationship: node-mutation site, size-occurrence count, edge-conservation
        # self._collect_mutation_info()  # collection mutation info and create table
        # self._plot_2D()  # 2D figure

        # substitution
        # self._boxplot_for_all_kinds()
        # self._boxplot_for_all_kinds("BA.4(Omicron)")
        # self._boxplot_for_all_kinds("B.1.617.2(Delta)")

        # variant
        # self._group_plot_with_node()
        self._group_plot_with_node_for_variant("BA.4(Omicron)")
        # self._group_plot_with_node_for_variant("B.1.617.2(Delta)")

    def _group_plot_with_node_for_variant(self, target_variant):
        # variant site
        variant_name = self.analysis_mutation_group.aa_groups_info["name"].to_list()
        variant_index = variant_name.index(target_variant)
        variant = self.analysis_mutation_group.aa_groups[variant_index]
        variant = [self._aa2position(i) for i in variant]
        # group sample data
        group_count_sample = self.analysis_mutation_group.group_count_sample

        # init figure
        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(2, 4, figsize=(14, 8), constrained_layout=True)
        axes = [j for i in axes for j in i]

        # draw
        funcs = self.get_functions()
        for index, (name, func) in enumerate(funcs.items()):
            # score
            variant_scores = func(variant)

            log.info("length is %s", len(variant_scores))
            grp_sample = group_count_sample[len(variant)]
            _sample_scores = [func(group) for group in grp_sample]
            sample_mean_score = [np.mean(group) for group in _sample_scores]
            sample_mean_score = sorted(sample_mean_score)  # 排序
            sample_scores = pd.Series(sample_mean_score).dropna().tolist()

            # draw
            variant_name = target_variant
            no_variant_name = "non-mutation"
            # box plot
            ax = axes[index]
            _plot_data = pd.DataFrame(
                {"score": variant_scores + sample_scores,
                 "label": [variant_name] * len(variant_scores) + [
                     no_variant_name] * len(sample_scores)}
            )
            x = "label"
            y = "score"
            order = [variant_name, no_variant_name]
            sns.boxplot(data=_plot_data, x=x, y=y, ax=ax, order=order, fliersize=1, width=.5)
            log.debug("variant score: %s", sorted(variant_scores))
            # tag p value
            self.boxplot_add_p_value(_plot_data, ax, x, y, order, "Mann-Whitney")
            mwu_2 = mannwhitneyu(variant_scores, sample_scores, alternative="two-sided")
            mwu_less = mannwhitneyu(variant_scores, sample_scores, alternative="less")
            mwu_greader = mannwhitneyu(variant_scores, sample_scores, alternative="greater")
            log.info("Mannwhitneyu result %s: for 2-side %s, for less %s, for greater %s", name, mwu_2, mwu_less,
                     mwu_greader)

            ax.set_xlabel("")
            ax.set_ylabel(name)

        # save fig
        fig_file_name = os.path.join(self.data_dir, target_variant, "boxplot_of_group.png")
        if not os.path.exists(os.path.dirname(fig_file_name)):
            os.mkdir(os.path.dirname(fig_file_name))

        fig.savefig(fig_file_name)

    def output_for_gephi(self):
        # edge
        rows = []
        for source, target in self.G.edges:
            is_neighbour = abs(int(target[:-1]) - int(source[:-1])) == 1
            is_mutation = (source in self.analysis_mutation_group.non_duplicated_aas_positions) or (
                    target in self.analysis_mutation_group.non_duplicated_aas_positions)
            if is_mutation:
                tag = "mutation"
            else:
                tag = "normal"

            w = self.G.edges[source, target]['weight']
            if is_neighbour and w < 0.9:
                w = 0.9

            rows.append([source, target, is_neighbour, is_mutation, tag, w])
        rows = pd.DataFrame(rows, columns=["source", "target", "is_neighbour", "is_mutation", "tag", "weight"], )
        rows.to_csv(os.path.join(self.data_dir, "network_edge_info.csv"), index=None)

        # nodes
        nodes = []
        for n in self.G.nodes:
            if n in self.analysis_mutation_group.non_duplicated_aas_positions:
                size = self.G.nodes[n]["size"]
                size = 0.001 if size == 0 else size
                node = [n, n, True, 5 + 1 / size]
            else:
                node = [n, "", False, 1]
            nodes.append(node)
        nodes = pd.DataFrame(nodes, columns=["Id", "Label", "is_mutation", "new_size"])
        nodes.to_csv(os.path.join(self.data_dir, "network_node_info.csv"), index=None)

    def output_for_DynaMut2(self):
        for i, (group, name) in enumerate(zip(self.analysis_mutation_group.aa_groups,
                                              self.analysis_mutation_group.aa_groups_info["name"]
                                              )):
            name = name.replace("/", "or")
            with open(os.path.join(self.data_dir, f"dynamut2 input v4 ({i} {name}).txt"), "w") as f:
                for a1, a2 in combinations(group, 2):
                    if "X" in a1 or "X" in a2:
                        continue
                    if name == "Omicron":
                        if self.G.has_edge(self._aa2position(a1), self._aa2position(a2)):
                            f.write(f"A {a1};A {a2}\n")
                    else:
                        f.write(f"A {a1};A {a2}\n")

    def random_sample_analysis(self, aas: list, groups, N=1000):
        aas = [self._aa2position(aa) for aa in aas if aa]
        aas = list(set(aas))
        groups = [[self._aa2position(aa) for aa in group] for group in groups]
        log.debug("aas = %s", aas)

        # resample
        group_and_sample_groups = []
        positions = pd.Series(self.positions)
        log.info("count %s", N)
        for group in groups:
            sample_groups = [positions.sample(n=len(group), random_state=i).tolist() for i in range(N)]
            group_and_sample_groups.append([group, sample_groups])

        def apply_to_group(data, func):
            scores = []
            for group, sample_groups in data:
                s1 = func(group)
                s2s = [func(g) for g in sample_groups]
                scores.append([s1, s2s])
            return scores

        # degree
        # weighted_degrees = self.get_weighted_degree()
        # avg_weighted_degrees = self.get_avg_weighted_degree()

        def calculate_degree(group: List[str]):
            data = []
            for aa in group:
                degree = self.G.degree[aa]
                data.append(degree)

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
        # data for plot
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
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        type2_count.plot(ax=ax)
        # ax.set_xtick(np.arange(max(type2)))
        ax.set_title("")
        ax.set_ylabel("density")
        fig.tight_layout()
        plt.show()
        fig.savefig(os.path.join(self.data_dir, "共保守性分布情况.png"), dpi=300)

    def _plot_mutations_relationship(self):
        groups = self.analysis_mutation_group.aa_groups
        aas = [self.analysis_mutation_group.aa2position(aa) for group in groups for aa in group]
        # edge
        # unique_aas = np.unique(aas)
        # node, weight which is the count
        aas_count = pd.value_counts(aas)
        # edge: whether has co-conservation
        edges = []
        for n1, n2 in combinations(aas_count.index, 2):
            if self.G.has_edge(n1, n2):
                edges.append([n1, n2, self.G.edges[n1, n2]["weight"]])
        # output
        # nodes = pd.DataFrame({"ID": aas_count.index, "weight": aas_count.values})
        # nodes.index.name = "rank"
        # links = pd.DataFrame(links, columns=["Source", "Target", "weight"])
        # links.index.name = "rank"
        #
        # nodes.to_csv(os.path.join(self.data_dir, "mutation relationship.gepi.node.csv"))
        # links.to_csv(os.path.join(self.data_dir, "mutation relationship.gepi.link.csv"))

        # # with networkx
        # log.debug("aas_count = %s", aas_count.shape)
        # log.debug("links = %s", len(links))
        # G = nx.Graph()
        # for node, weight in aas_count.items():
        #     G.add_node(node, weight=weight)
        # for n1, n2, weight in links:
        #     G.add_edge(n1, n2, weight=weight)
        # fig = plt.figure(figsize=(20, 20), )
        # nx.draw_circular(G, )
        # nx.draw_networkx_labels(G, nx.drawing.circular_layout(G, scale=1.05), font_size=3, font_color="g", )
        # fig.tight_layout()
        # fig.show()
        # fig.savefig(os.path.join(self.data_dir, "mutation count relationships.png"), dpi=300)

        # with echart
        nodes = pd.DataFrame({"name": aas_count.index, "symbolSize": aas_count.values})
        nodes = nodes.to_dict("records")
        links = pd.DataFrame(edges, columns=["source", "target", "weight"])
        links.to_csv(os.path.join(self.data_dir, "test_links.csv"))
        links = links.to_dict("records")
        count_node_in_links = defaultdict(int)
        for link in links:
            source = link["source"]
            target = link["target"]
            count_node_in_links[source] += 1
            count_node_in_links[target] += 1
        log.debug("pd.Series(count_node_in_links) = %s", pd.Series(count_node_in_links).sort_values())

        c = (
            Graph(init_opts=opts.InitOpts(width="100%", height="1000px"))
            .add("", nodes, links, repulsion=8000, layout="circular", )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="count"),
                toolbox_opts=opts.ToolboxOpts(
                    feature=opts.ToolBoxFeatureOpts(
                        save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(pixel_ratio=3, background_color="white"))), )
            .render(os.path.join(self.data_dir, "mutation relationship.html"))
        )

    def _boxplot_for_all_kinds(self, target_variant=None):
        def _func_boxplot(variant, sample, ax, func, name, target_variant=None):
            if not target_variant is None:
                target_variant = re.findall("\((.*?)\)", target_variant)[0]
            variant_scores = func(variant)
            sample_scores = [j for i in sample for j in func(i)]

            variant_name = "mutation" if target_variant is None else target_variant
            no_variant_name = "non-mutation"
            # box plot
            _plot_data = pd.DataFrame(
                {"score": variant_scores + sample_scores,
                 "label": [variant_name] * len(variant_scores) + [
                     no_variant_name] * len(sample_scores)}
            )
            x = "label"
            y = "score"
            order = [variant_name, no_variant_name]
            sns.boxplot(data=_plot_data, x=x, y=y, ax=ax, order=order, fliersize=1, width=.5)
            # tag p value
            self.boxplot_add_p_value(_plot_data, ax, x, y, order, "Mann-Whitney")
            mwu_2 = mannwhitneyu(variant_scores, sample_scores, alternative="two-sided")
            mwu_less = mannwhitneyu(variant_scores, sample_scores, alternative="less")
            mwu_greader = mannwhitneyu(variant_scores, sample_scores, alternative="greater")
            log.info("Mannwhitneyu result %s: for 2-side %s, for less %s, for greater %s", name, mwu_2, mwu_less,
                     mwu_greader)

            ax.set_xlabel("")
            ax.set_ylabel(name)

        # init figure
        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(2, 4, figsize=(14, 8), constrained_layout=True)
        axes = [j for i in axes for j in i]
        # init func
        funcs = self.get_functions()

        # init data
        if target_variant is None:
            variant = self.analysis_mutation_group.non_duplicated_aas_positions
        else:
            variant_name = self.analysis_mutation_group.aa_groups_info["name"].to_list()
            variant_index = variant_name.index(target_variant)
            variant = self.analysis_mutation_group.aa_groups[variant_index]
            variant = [self._aa2position(i) for i in variant]
        sample = self.analysis_mutation_group.non_duplicated_aas_sample

        for idx, (key, func) in enumerate(funcs.items()):
            ax = axes[idx]
            _func_boxplot(variant, sample, ax, func, key, target_variant=target_variant)
        fig.show()

        # save fig
        if target_variant is None:
            fig_file_name = os.path.join(self.data_dir,
                                         "Figure 4 Comparison between variant nodes and sampled nodes on network characteristics.png")
        else:
            fig_file_name = os.path.join(self.data_dir, target_variant, "boxplot_of_all.png")
            if not os.path.exists(os.path.dirname(fig_file_name)):
                os.mkdir(os.path.dirname(fig_file_name))

        fig.savefig(fig_file_name)

    def get_functions(self):
        funcs = {
            "CS": self.calculate_conservation,
            "CCS": self.calculate_co_conservation,
            "Kw": self.calculate_avg_weighted_degree,
            "P": self.calculate_page_rank,
            "D": self.calculate_degree_centrality,
            "B": self.calculate_betweenness_centrality,
            "C": self.calculate_closeness_centrality,
            "L": self.calculate_weighted_shortest_path,
        }
        return funcs

    def _group_plot_with_node(self):
        groups = self.analysis_mutation_group.aa_groups_position
        groups_names = self.analysis_mutation_group.aa_groups_info["name"]
        groups_colors = self.analysis_mutation_group.aa_groups_info["color"]
        group_count_sample = self.analysis_mutation_group.group_count_sample

        def calculate_group_and_sample_score(grp, grp_sample, func, fig_name, kind="distribution", excel_writer=None):
            # score
            grp_scores = [func(i) for i in grp]
            grp_mean_score = [np.mean(i) for i in grp_scores]
            grp_sample_scores = {}
            for count, sample_group in grp_sample.items():
                sample_scores = [func(group) for group in sample_group]
                sample_mean_score = [np.mean(group) for group in sample_scores]
                sample_mean_score = sorted(sample_mean_score)  # 排序
                grp_sample_scores[count] = sample_mean_score

            grp_info = pd.DataFrame({"name": groups_names, "score": grp_mean_score, "color": groups_colors})
            grp_info["length"] = [len(g) for g in grp]
            grp_info["index"] = np.arange(len(grp_info)) + 1

            # draw
            if kind == "distribution":
                count_plot = len(grp_sample_scores)
                fig: plt.Figure = plt.figure(figsize=(20, 30), )
                axes: List[plt.Axes] = fig.subplots(math.ceil(count_plot / 3), 3, )
                colors = sns.color_palette(n_colors=len(grp_sample_scores))
                axes = [j for i in axes for j in i]
                ax_all_in_one = axes[count_plot]
                ax_box_plot = axes[count_plot + 1]
                [ax.clear() for ax in axes[count_plot + 2:]]

                statistic_table = []
                for i, N in enumerate(grp_sample_scores.keys()):
                    # draw distribution plot
                    color = colors[i]
                    ax: plt.Axes = axes[i]
                    sample_mean_score = grp_sample_scores[N]
                    # sns.distplot(sample_mean_score, ax=ax, color=color)
                    # sns.distplot(sample_mean_score, ax=ax_all_in_one, color=color)

                    sns.histplot(sample_mean_score, ax=ax, color=color, kde=True, stat="probability")
                    sns.histplot(sample_mean_score, ax=ax_all_in_one, color=color, kde=True, stat="probability")

                    # statistic data
                    statistic_data = {
                        "a quarter of a quintile": np.percentile(sample_mean_score, 25),
                        "one half quantile": np.percentile(sample_mean_score, 50),
                        "three quarters of a quintile": np.percentile(sample_mean_score, 75),
                        "mean": np.mean(sample_mean_score)
                    }

                    # analysis by N
                    current_n_grp_info = grp_info[grp_info["length"] == N]
                    current_n_grp_info = current_n_grp_info.sort_values("score", ascending=False)
                    for index, row in current_n_grp_info.iterrows():
                        _x = [row["score"]] * len(sample_mean_score)
                        _y = range(1, len(sample_mean_score) + 1)
                        ax.axvline(row["score"], ls="-", label=row["name"], color=row["color"])
                        ax_all_in_one.axvline(row["score"], ls="-", label=row["name"], color=row["color"])

                        # collect data
                        t, pvalue = ttest_1samp(sample_mean_score, row["score"])
                        if pvalue < 0.05:
                            if t < 0:
                                is_remarkable = "Yes. Bigger."
                            else:
                                is_remarkable = "Yes. Smaller."
                        else:
                            is_remarkable = "No"

                        statistic_data["score"] = row["score"]
                        statistic_data["t"] = t
                        statistic_data["p"] = pvalue
                        statistic_data["result"] = is_remarkable
                        statistic_data["name"] = row["name"]
                        statistic_data["index"] = row["index"]

                        statistic_table.append(statistic_data.copy())

                    ax.set_title(f"N = {N}", y=-0.1)
                    ax.legend()
                ax_all_in_one.set_title(f"all", y=-0.1)
                # ax_all_in_one.legend()
                # ax_all_in_one.get_legend().remove()

                # boxplot
                x = "label"
                y = "score"
                order = ["variant", "non-variant"]

                _s1 = grp_mean_score
                _s2 = np.array(list(grp_sample_scores.values())).reshape(-1).tolist()
                _plot_data = pd.DataFrame(
                    {"score": _s1 + _s2, "label": ["variant"] * len(_s1) + ["non-variant"] * len(_s2)},
                )
                sns.boxplot(data=_plot_data, x=x, y=y, ax=ax_box_plot, order=order, fliersize=1, width=0.5)
                self.boxplot_add_p_value(_plot_data, ax_box_plot, x, y, order, )
                ax_box_plot.set_xlabel("")
                ax_box_plot.set_ylabel(fig_name)

                global_axes = self.group_global_axes[self.group_global_ax_count]
                sns.boxplot(data=_plot_data, x=x, y=y, order=order, ax=global_axes, fliersize=1, width=0.5)
                self.boxplot_add_p_value(_plot_data, global_axes, x, y, order, )
                global_axes.set_xlabel("")
                global_axes.set_ylabel(fig_name)
                self.group_global_ax_count += 1

                # # global valid
                # if p_value <= 0.05:
                #     global_axes = self.group_global_valid_axes[self.group_global_valid_ax_count]
                #     self.group_global_valid_ax_count += 1
                #     sns.boxplot(data=_plot_data, x="label", y="score", ax=global_axes, fliersize=1)
                #     # self.group_global_valid_fig.show()
                #     # global_axes.set_xlabel(f"{fig_name} (p={p_value:.3f})", y=-0.1)
                #     global_axes.set_title(f"p = {p_value:.3f}", y=0.9)
                #     global_axes.set_xlabel("")
                #     global_axes.set_ylabel(fig_name)

                # output
                # fig.suptitle(fig_name, )
                fig.tight_layout()
                fig.show()

                # save
                group_plot_basedir = os.path.join(self.data_dir, "group plot")
                if not os.path.exists(group_plot_basedir):
                    os.mkdir(group_plot_basedir)
                fig.savefig(os.path.join(group_plot_basedir,
                                         f"Supplemental Figure {self.group_global_ax_count}. group distribution {fig_name}.png"),
                            dpi=300)

                # table
                if excel_writer:
                    statistic_table = pd.DataFrame(statistic_table)
                    statistic_table = statistic_table.sort_values("index")
                    statistic_table.to_excel(excel_writer, sheet_name=fig_name)

            elif kind == "score_sorted":
                fig: plt.Figure = plt.figure(figsize=(20, 25))
                axes: List[plt.Axes] = fig.subplots(3, 3, )
                axes = [j for i in axes for j in i]
                ax_all_in_one = axes[7]
                # _s2 = []
                for i, N in enumerate(grp_sample_scores.keys()):
                    ax: plt.Axes = axes[i]
                    sample_mean_score = grp_sample_scores[N]
                    # _s2 += grp_info[grp_info["length"] == N].shape[0] * np.array(sample_mean_score).reshape(-1).tolist()
                    ax.plot(range(1, len(sample_mean_score) + 1), sample_mean_score)
                    ax_all_in_one.plot(range(1, len(sample_mean_score) + 1), sample_mean_score)
                    for index, row in grp_info[grp_info["length"] == N].sort_values("score",
                                                                                    ascending=False).iterrows():
                        ax.plot(range(1, len(sample_mean_score) + 1), [row["score"]] * len(sample_mean_score),
                                label=row["name"])
                        ax_all_in_one.plot(range(1, len(sample_mean_score) + 1),
                                           [row["score"]] * len(sample_mean_score),
                                           label=row["name"])
                        # ax.loglog(range(1, len(sample_mean_score) + 1), [row["score"]] * len(sample_mean_score))
                        # ax.text(x=0, y=row["score"], s=row["name"])

                    ax.set_title(f"N = {N}", y=-0.1)
                    ax.legend()
                ax_all_in_one.set_title(f"all", y=-0.1)
                _s1 = grp_mean_score
                _s2 = np.array(list(grp_sample_scores.values())).reshape(-1).tolist()
                _plot_data = pd.DataFrame(
                    {"score": _s1 + _s2, "label": ["mutation"] * len(_s1) + ["sample"] * len(_s2)},
                )
                sns.boxplot(data=_plot_data, x="label", y="score", ax=axes[-1])

                p_value = mannwhitneyu(_s1, _s2).pvalue
                axes[-1].set_title(f"p = {p_value:.3f}", y=-0.1)
                axes[-1].set_xlabel("")

                fig.suptitle(fig_name, )
                fig.tight_layout()
                fig.show()
                fig.savefig(os.path.join(self.data_dir, f"group {fig_name}.png"), dpi=300)

        # boxplot
        self.group_global_fig: plt.Figure = plt.figure(figsize=(14, 8))
        self.group_global_axes = [j for i in self.group_global_fig.subplots(2, 4) for j in i]
        self.group_global_ax_count = 0
        # p < 0.05
        self.group_global_valid_fig: plt.Figure = plt.figure(figsize=(16, 8))
        self.group_global_valid_axes = [j for i in self.group_global_valid_fig.subplots(3, 3) for j in i]
        self.group_global_valid_ax_count = 0

        # table
        excel_writer = pd.ExcelWriter(os.path.join(self.data_dir, "group distribution statistic information.xlsx"))

        funcs = self.get_functions()
        for name, func in funcs.items():
            calculate_group_and_sample_score(groups, group_count_sample, func, name, excel_writer=excel_writer)

        self.group_global_fig.tight_layout()
        self.group_global_fig.savefig(os.path.join(self.data_dir,
                                                   "Figure 5 Comparison between variants and samples on network characteristics.png"),
                                      dpi=300)
        [i.set_visible(False) for i in self.group_global_valid_axes[self.group_global_valid_ax_count:]]  # 删除多余子图
        self.group_global_valid_fig.tight_layout()
        self.group_global_valid_fig.savefig(os.path.join(self.data_dir, "group distribution global valid.png"), dpi=300)

        excel_writer.close()

    def calculate_degree_centrality(self, grp):
        dc = scaler(self.degree_c)
        return [dc[aa] for aa in grp]

    def calculate_betweenness_centrality(self, grp):
        bc = scaler(self.betweenness_c)
        return [bc[aa] for aa in grp]

    def calculate_closeness_centrality(self, grp):
        cc = scaler(self.closeness_c)
        return [cc[aa] for aa in grp]

    def calculate_degree(self):
        result = {}
        for node in self.G.nodes:
            weighted_degrees = []
            for nbr, datadict in self.G.adj[node].items():
                weighted_degrees.append(datadict.get("weight", 0))

            if weighted_degrees:
                avg_weighted_degree = np.mean(weighted_degrees)
            else:
                avg_weighted_degree = 0
            result[node] = avg_weighted_degree
        return result

    def calculate_avg_weighted_degree(self, grp):
        degree = scaler(self.degree)
        return [degree[aa] for aa in grp]

    def calculate_page_rank(self, grp):
        pr = scaler(self.page_rank)
        return [pr[aa] for aa in grp]

    def calculate_conservation(self, grp):
        return [self.G.nodes[aa]["size"] for aa in grp]

    def get_shortest_path_length(self):
        if not os.path.exists(os.path.join(self.data_dir, "weighted shortest path length.json")):
            weighted_shortest_path_length = dict(nx.shortest_path_length(self.G, weight="weight"))
            with open(os.path.join(self.data_dir, "weighted shortest path length.json"), "w") as f:
                log.info("save")
                f.write(json.dumps(weighted_shortest_path_length))
        else:
            with open(os.path.join(self.data_dir, "weighted shortest path length.json"), ) as f:
                log.info("load")
                weighted_shortest_path_length = json.loads(f.read())
        return weighted_shortest_path_length

    def calculate_weighted_shortest_path(self, grp, is_dict=False):
        res = []
        names = []
        for n1, n2 in combinations(grp, 2):
            res.append(self.shortest_path_length[n1][n2])
            if is_dict:
                names.append(f"{n1}-{n2}")

        if is_dict:
            return dict(zip(names, res))
        else:
            return res

    def calculate_co_conservation(self, grp, is_dict=False):
        res = []
        names = []
        for n1, n2 in combinations(grp, 2):
            if self.G.has_edge(n1, n2):
                res.append(self.G.edges[n1, n2]["weight"])
                if is_dict:
                    names.append(f"{n1}-{n2}")
        if is_dict:
            return dict(zip(names, res))
        else:
            return res

    def calculate_edge_betweenness_centrality(self, grp):
        # if not hasattr(self, "edge_betweenness_centrality"):
        #     try:
        #         # 保存至 json 文件
        #         if not os.path.exists(os.path.join(self.data_dir, "edge betweenness centrality.json")):
        #             log.info("没有 self.edge_betweenness_centrality, 初始化")
        #             self.edge_betweenness_centrality = dict(nx.edge_betweenness_centrality(self.G, weight="weight"))
        #             # 重新建立索引
        #             info_map = defaultdict(dict)
        #             for (n1, n2), value in self.edge_betweenness_centrality.items():
        #                 info_map[n1][n2] = value
        #                 info_map[n2][n1] = value
        #
        #             with open(os.path.join(self.data_dir, "edge betweenness centrality.json"), "w") as f:
        #                 log.info("保存至文件")
        #                 f.write(json.dumps(info_map))
        #
        #         with open(os.path.join(self.data_dir, "edge betweenness centrality.json"), ) as f:
        #             log.info("加载文件")
        #             self.edge_betweenness_centrality = json.loads(f.read())
        #     except:
        #         log.error("保存或加载文件失败")
        #     log.debug("初始化完成")
        #
        # ebc = self.edge_betweenness_centrality
        # # ebc_values = []
        # # for k1, v1 in ebc.items():
        # #     for k2, v2 in v1.items():
        # #         ebc_values.append(v2)
        # # scaler = preprocessing.MinMaxScaler().fit(np.array(ebc_values))
        #
        # for n1, n2 in combinations(grp, 2):
        #     if n1 in ebc and n2 in ebc[n1]:
        #         res.append(ebc[n1][n2])
        #     elif n2 in ebc and n1 in ebc[n2]:
        #         res.append(ebc[n2][n1])
        #     else:
        #         res.append(0)
        if not hasattr(self, "edge_betweenness_centrality_scaler"):
            self.edge_betweenness_centrality_scaler = scaler(self.edge_betweenness_c)

        ebc = self.edge_betweenness_centrality_scaler
        res = []
        for n1, n2 in combinations(grp, 2):
            if (n1, n2) in self.edge_betweenness_c:
                res.append(ebc[(n1, n2)])
            elif (n2, n1) in self.edge_betweenness_c:
                res.append(ebc[(n2, n1)])
            # else:
            #     res.append(0)
        return res

    def calculate_co_conservation_rate(self):
        # find pairwise, create index
        # type2 = self.type2[self.type2["info"] >= self.threshold]
        type2 = self.type2
        idx = defaultdict(list)
        for i, row in type2.iterrows():
            idx[row.site1].append(row.site2)
            idx[row.site2].append(row.site1)

        def _cal(grp):
            rns = len(grp)
            rnp = comb(rns, 2)
            rpc = 0
            for p1, p2 in combinations(grp, 2):
                if p2 in idx[p1]:
                    rpc += 1
            res = [rpc / rnp, ] * len(grp)
            return res

        return _cal

    def _plot_2D(self, font_size="x-large", txt_rotation=0, x_rotation=90):
        analysis = self.analysis_mutation_group.analysis
        type1 = []
        type2 = []
        name2index = {}
        co_conservation_rate = []
        for i, (_, row) in enumerate(analysis.items()):
            t1 = pd.DataFrame(row["type1"])
            t1["y"] = i

            if type(row["WHO label"]) == str:
                name = "{}({})".format(row["Lineage + additional mutations"], row["WHO label"], )
            else:
                name = "{}".format(row["Lineage + additional mutations"], )
            t1["name"] = name
            name2index[name] = i

            # type2
            t2 = row["type2"]
            t2 = pd.DataFrame(t2)
            t2["y1"] = t2["y2"] = i

            type1.append(t1)
            type2.append(t2)

            rns = len(t1)
            rnp = comb(rns, 2)
            # rpc = len(t2)
            rpc = (t2["info"] >= self.threshold).sum() if "info" in t2.columns else 0
            rr = rpc / rnp
            co_conservation_rate.append(
                {
                    "Name": name,
                    "Count(substitution)": rns,
                    "Count(pairwise)": rnp,
                    "Count(pairwise with co-conservation)": rpc,
                    "Rate": rr
                }
            )
        co_conservation_rate = pd.DataFrame(co_conservation_rate)
        co_conservation_rate.to_csv(
            os.path.join(self.data_dir, "Supplemental Table 2. Rate of pairwise with co-conservation.csv")
        )

        type1 = pd.concat(type1)
        type2 = pd.concat(type2)
        type2 = type2[type2["info"] >= self.threshold]

        """
        4 part
            * x 
            * y
            * node
            * edge
        """

        # x
        type1["idx"] = type1["aa"].str[1:-1].astype(int)
        pst_2_x = type1["idx"].drop_duplicates().sort_values().to_list()
        pst_2_x = {j: i for i, j in enumerate(pst_2_x)}
        type1["x"] = type1["idx"].map(pst_2_x)
        type2["x1"] = type2["site1"].str[:-1].astype(int).map(pst_2_x)
        type2["x2"] = type2["site2"].str[:-1].astype(int).map(pst_2_x)

        gap = 1
        type2["x3"] = (type2["x1"] + type2["x2"]) / 2
        log.debug("type2 = %s", type2)
        type2["y3"] = (type2["y1"] + gap / 4) + (type2["x2"] - type2["x1"]).abs() / (len(pst_2_x) - 2)

        # # edge
        # type2_info = type2.loc[:, ["site1", "site2", "info", "rate"]].drop_duplicates(["site1", "site2"])
        # log.debug("type2_info = %s", type2_info)

        # draw
        def two_degree_bc(x, y, dots_num=100):  # bezier curve
            x1, x2, x3 = x
            y1, y2, y3 = y
            xt = []
            yt = []
            x_dots12 = np.linspace(x1, x2, dots_num)
            y_dots12 = np.linspace(y1, y2, dots_num)
            x_dots23 = np.linspace(x2, x3, dots_num)
            y_dots23 = np.linspace(y2, y3, dots_num)  # 线段BC的y坐标
            for i in range(dots_num):
                x = x_dots12[i] + (x_dots23[i] - x_dots12[i]) * i / (dots_num - 1)
                y = y_dots12[i] + (y_dots23[i] - y_dots12[i]) * i / (dots_num - 1)
                xt.append(x)
                yt.append(y)
            xt = np.array(xt)
            yt = np.array(yt)
            return xt, yt

        # flg_size = [max(len(pst_2_x), len(analysis)), ] * 2
        # fig_size = np.array(flg_size) / 2.6
        fig_size = (28, 14)
        print("fig_size:", fig_size)
        flg, ax = plt.subplots(figsize=fig_size)

        # text
        for i, row in type1.sort_values(["y", ]).iterrows():
            x, y, txt = row["x"], row["y"], row["aa"]
            ax.text(x, y, txt[-1], ha="center", va="center", size=font_size, rotation=txt_rotation)

        # curve
        for i, row in type2.iterrows():
            x1, x2, x3, y1, y2, y3 = row[["x1", "x2", "x3", "y1", "y2", "y3"]]
            x = [x1, x3, x2]
            y = [y1, y3, y2]
            x_smooth, y_smooth = two_degree_bc(x, y)
            y_smooth += 0.15
            alpha = row["rate"] / 100
            ax.plot(x_smooth, y_smooth, "C1", alpha=alpha * 2)
            # ax.plot(x_smooth, y_smooth, "C1",)

        flg.show()
        # tag
        xtick_names = type1["position"].drop_duplicates()
        arg_sort = xtick_names.str[:-1].astype(int).argsort()
        xtick_names = xtick_names.iloc[arg_sort]
        ytick_names = type1["name"].drop_duplicates()
        arg_sort = ytick_names.apply(lambda x: name2index[x]).astype(int).argsort()
        ytick_names = ytick_names.iloc[arg_sort]

        ax.set_xticks(range(len(xtick_names)))
        ax.set_yticks(range(len(ytick_names)))
        ax.set_xticklabels(xtick_names, size=font_size)
        ax.set_yticklabels(ytick_names, size=font_size)
        # ax.invert_yaxis()
        # ax.set_ymargin(0.03)
        # ax.set_xmargin(-0.1)
        ax.margins(0.02)
        [txt.set_rotation(x_rotation) for txt in ax.get_xticklabels()]

        # save
        flg.tight_layout()
        flg.show()
        flg.savefig(os.path.join(self.data_dir, "2D relationship.png"), dpi=300)

    def output_for_topnettree(self):
        # find no duplicated
        log.info(self.analysis_mutation_group.non_duplicated_aas)

    def generate_ebc(self):
        if not hasattr(self, "edge_betweenness_centrality"):
            try:
                # save
                if not os.path.exists(os.path.join(self.data_dir, "edge betweenness centrality.json")):
                    log.info("no self.edge_betweenness_centrality, init")
                    self.edge_betweenness_centrality = dict(nx.edge_betweenness_centrality(self.G, weight="weight"))
                    info_map = defaultdict(dict)
                    for (n1, n2), value in self.edge_betweenness_centrality.items():
                        info_map[n1][n2] = value
                        info_map[n2][n1] = value

                    with open(os.path.join(self.data_dir, "edge betweenness centrality.json"), "w") as f:
                        log.info("save")
                        f.write(json.dumps(info_map))

                with open(os.path.join(self.data_dir, "edge betweenness centrality.json"), ) as f:
                    log.info("load")
                    self.edge_betweenness_centrality = json.loads(f.read())
            except:
                log.error("error")
            log.debug("success")

    def boxplot_add_p_value(self, df, ax, x, y, order, test='Mann-Whitney'):
        annot = Annotator(ax, [order, ], data=df, x=x, y=y, order=order)
        annot.configure(test=test, text_format='star', loc='inside', verbose=2)
        annot.apply_test().annotate()
        ax.set_title("")
        return ax

    def generate_all_node_top_info(self, top=10):
        seq = self.analysis_mutation_group.fasta
        aas = []
        for i, aa in enumerate(seq):
            i = i + 1
            aa = f"{i}{aa}"
            aas.append(aa)

        # cal
        funcs = self.get_functions()
        node_info = {"aas": aas}
        pair_info = {}
        for fname, func in funcs.items():
            if fname in ["CCS", "L"]:
                pair_info[fname] = func(aas, is_dict=True)
            else:
                node_info[fname] = func(aas)

        # node info
        node_info = pd.DataFrame(node_info)
        node_info = node_info.set_index("aas")
        node_info.to_csv(os.path.join(self.data_dir, "node_info.csv"))
        # find top
        node_info_top = {}
        for label, content in node_info.iteritems():
            content_sorted = content.sort_values(ascending=False)
            node_info_top[label] = content_sorted.index.to_list()[:top]
        node_info_top = pd.DataFrame(node_info_top)
        node_info_top.to_csv(os.path.join(self.data_dir, "node_info_top.csv"))

        # pair info
        pair_info_top = {}
        for lable, content in pair_info.items():
            content = pd.Series(content)
            content_sorted = content.sort_values(ascending=False)
            pair_info_top[lable] = content_sorted.index.to_list()[:top]
        pair_info_top = pd.DataFrame(pair_info_top)
        pair_info_top.to_csv(os.path.join(self.data_dir, "pair_info_top.csv"))

        # variation node
        mutations = self.analysis_mutation_group.non_duplicated_aas_positions
        mutations = pd.Series(mutations)
        mutations.to_csv(os.path.join(self.data_dir, "mutation positions.csv"), header=None)


if __name__ == '__main__':
    start_time = time.time()
    # conservation network
    # important variant
    mutation_groups = AnalysisMutationGroup()
    mutation_groups.display_seq_and_aa()
    mutation_groups.count_aa()
    pcn = ProConNetwork(mutation_groups, threshold=100)
    pcn.analysisG()  # draw

    # pcn.generate_all_node_top_info()

    # pcn._collect_mutation_info()  # save info

    # generate data for other tools
    # pcn.generate_ebc()
    # pcn.output_for_gephi()
    # pcn.output_for_DynaMut2()
    # pcn.output_for_topnettree()
    end_time = time.time()
    log.info(f"run time: {end_time - start_time}")
    # print(mutation_groups.non_duplicated_aas)

    # thresholds = [50, 100, 150, 200, 250, 300]
    # for t in thresholds:
    #     if t != 100:
    #         continue
    #     pcn = ProConNetwork(mutation_groups, threshold=t)
    #
    #     pcn.analysisG()
    #     # pcn.random_sample_analysis(aas, groups.get_aa_groups())
    #
    #     end_time = time.time()
    #     log.info(f"run time: {end_time - start_time}")
