from functools import reduce

import networkx as nx
import numpy as np
from Bio import SeqIO
from output_procon_analysis import *
import pickle
import re
import pandas as pd


class AllostericAnalysis():
    G_path = "../data/procon/threshold_100/G.pickle"
    fasta_path = "../data/YP_009724390.1.txt"

    def __init__(self, G):
        self.G = G
        self.allosteric_sites = self.load_allosteric_sites()
        self.result_path = "../data/allosteric_analysis_result.txt"
        self.result = open(self.result_path, "w")

    def close(self):
        self.result.close()

    @staticmethod
    def load_mutation_positions(path="../data/procon/threshold_100/mutation positions.csv"):
        mutations = pd.read_csv(path, header=None)
        mutations = mutations.iloc[:, 1]
        res = mutations.to_list()
        return res

    @staticmethod
    def load_network_top_positions(
            node_path="../data/procon/threshold_100/node_info_top.csv",
            pair_path="../data/procon/threshold_100/pair_info_top.csv"
    ):
        res = {}
        # node
        node_info = pd.read_csv(node_path, index_col=0)
        for label, content in node_info.iteritems():
            res[label] = content.to_list()

        # pair
        pair_info = pd.read_csv(pair_path, index_col=0)
        for label, content in pair_info.iteritems():
            pair = content.to_list()
            sites_of_pair = [i.split("-") for i in pair]
            sites_of_pair = list(reduce(lambda a, b: a + b, sites_of_pair))
            res[label] = sites_of_pair
        return res

    def analysis_top_site(self):
        top_info = self.load_network_top_positions()
        mutations = self.load_mutation_positions()
        # top sites
        # find overlap between top sites and mutations for different properties
        self.result.write("find overlap between top sites and mutations for different properties\n")
        for label, content in top_info.items():
            set1 = set(content)
            set2 = set(mutations)
            intersection = list(set1 & set2)
            intersection = sorted(intersection, key=lambda x: int(x[:-1]))
            msg = "{}: {}".format(label, intersection)
            self.result.write(msg + "\n")
        self.result.write("\n")

        # find overlap between top sites and alloteric sites
        self.result.write("find overlap between top sites and alloteric sites\n")
        for label, content in top_info.items():
            intersection = self.find_node_in_allosteric_sites(content)
            msg = "{}: {}".format(label, intersection)
            self.result.write(msg + "\n")
        self.result.write("\n")

        # find overlap between top sites neighbour and alloteric sites
        self.result.write("find overlap between top sites neighbour and allosteric sites\n")
        for label, content in top_info.items():
            self.result.write(label + "\n")
            for site in content:
                neighbor = list(nx.neighbors(self.G, site))
                intersection = self.find_node_in_allosteric_sites(neighbor)
                msg = "{}: {}\n".format(site, ", ".join(intersection))
                self.result.write(msg)
            self.result.write("\n")
        self.result.write("\n")

    @staticmethod
    def load_paper_sites(path="../data/PaperSite.txt", ):
        # with open(AllostericAnalysis.fasta_path) as f:
        seq = SeqIO.read(AllostericAnalysis.fasta_path, "fasta").seq
        psts = {i + 1: f"{i + 1}{j}" for i, j in enumerate(seq)}
        # print(psts)

        with open(path) as f:
            txt = f.read()
        sites = re.findall("\d+", txt)
        sites = list(set(sites))
        sites = list(map(int, sites))
        sites = [psts[i] for i in sites]

        return sites

    @staticmethod
    def load_allosteric_sites(path="../data/AllostericSite.txt"):
        with open(path) as f:
            txt = f.read()
        sites = re.findall("\d+", txt)
        sites = list(set(sites))
        sites = list(map(int, sites))
        return sites

    @staticmethod
    def dump_G():
        mutation_groups = AnalysisMutationGroup()
        mutation_groups.display_seq_and_aa()
        mutation_groups.count_aa()
        pcn = ProConNetwork(mutation_groups, threshold=100)
        log.info("存储 G %s", AllostericAnalysis.G_path)
        with open(AllostericAnalysis.G_path, "wb") as f:
            pickle.dump(pcn.G, f)

    @staticmethod
    def load_G():
        log.info("加载 G %s", AllostericAnalysis.G_path)
        with open(AllostericAnalysis.G_path, "rb") as f:
            G = pickle.load(f)
        return G

    def find_node_in_allosteric_sites(self, nodes):
        sp_site = [int(i[:-1]) for i in nodes]
        exit_site = list(set(sp_site) & set(self.allosteric_sites))
        exit_site = sorted(exit_site)
        seq = SeqIO.read(AllostericAnalysis.fasta_path, "fasta").seq
        psts = {i + 1: f"{i + 1}{j}" for i, j in enumerate(seq)}
        exit_site = [psts[i] for i in exit_site]

        log.info("site %s", exit_site)
        return exit_site

    def find_shortest_path(self, pst1, pst2):
        G = self.G
        sp = nx.shortest_path(G, source=pst1, target=pst2)
        sp_sorted = sorted(sp, key=lambda x: int(x[:-1]))
        log.info("shortest path(%s -> %s): %s", pst1, pst2, "->".join(sp))
        log.info("shortest path sorted(%s -> %s): %s", pst1, pst2, sp_sorted)
        self.find_node_in_allosteric_sites(sp)
        return sp

    def find_neighbour(self, n):
        G = self.G
        neighbours = list(nx.neighbors(G, n))
        log.info("%s neighbour: %s", n, neighbours)
        self.find_node_in_allosteric_sites(neighbours)

    def find_common_site_paper_important_site_and_allosteric_site(self):
        allosteric_sites = self.allosteric_sites
        paper_sites = self.load_paper_sites()
        log.info("allo sites: %s", allosteric_sites)
        log.info("paper sites: %s", paper_sites)
        res = self.find_node_in_allosteric_sites(paper_sites)
        self.result.write("paper important sites & allosteric sites\n")
        self.result.write("{}\n".format(res))
        self.result.write("\n")

        # mutations
        mutations = self.load_mutation_positions()
        res = self.find_node_in_allosteric_sites(mutations)
        self.result.write("all mutation sites & allosteric sites\n")
        self.result.write("{}\n".format(res))
        self.result.write("\n")

    @staticmethod
    def index_seq():
        seq = SeqIO.read(AllostericAnalysis.fasta_path, "fasta").seq
        res = pd.Series(list(seq), index=np.arange(len(seq)) + 1)
        res.to_csv("../data/seq_with_index.csv")
        with open("../data/AllostericSite.txt") as f:
            txt = f.read()
        sites = re.findall("\d+", txt)
        sites = list(map(int, sites))
        check_sites = pd.DataFrame({
            "Site": sites,
            "Real AA": res.loc[sites]
        })
        check_sites.to_excel("../data/check_allosteric_site.xlsx")
        print(res[sites])


if __name__ == '__main__':
    # AllostericAnalysis.dump_G()
    g = AllostericAnalysis.load_G()
    AllostericAnalysis.index_seq()

    # # find average shortest path length
    # analysis = AllostericAnalysis(g)
    # analysis.analysis_top_site()
    # analysis.find_common_site_paper_important_site_and_allosteric_site()
    # # BEF L
    # bfe_l = [("493Q", "547T"), ("478T", "493Q"), ("213V", "493Q")]
    # for pst1, pst2 in bfe_l:
    #     log.info("nodes in SP %s-%s", pst1, pst2)
    #     analysis.find_shortest_path(pst1, pst2)
    #
    # for pst1, pst2 in bfe_l:
    #     log.info("neighbour %s", pst1)
    #     analysis.find_neighbour(pst1)
    #     log.info("neighbour %s", pst2)
    #     analysis.find_neighbour(pst2)
    #
    # analysis.close()
