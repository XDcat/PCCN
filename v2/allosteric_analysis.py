import networkx as nx
from Bio import SeqIO
from output_procon_analysis import *
import pickle
import re


class AllostericAnalysis():
    G_path = "../data/procon/threshold_100/G.pickle"
    fasta_path = "../data/YP_009724390.1.txt"

    def __init__(self, G):
        self.G = G
        self.allosteric_sites = self.load_allosteric_sites()

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
        log.info("相交位点 %s", exit_site)

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

    def find_common_site(self):
        allosteric_sites = self.allosteric_sites
        paper_sites = self.load_paper_sites()
        log.info("allo sites: %s", allosteric_sites)
        log.info("paper sites: %s", paper_sites)
        log.info("全文相交节点 ")
        self.find_node_in_allosteric_sites(paper_sites)



if __name__ == '__main__':
    # AllostericAnalysis.dump_G()
    g = AllostericAnalysis.load_G()

    # 寻找平均最短路径
    analysis = AllostericAnalysis(g)
    analysis.find_common_site()
    # BEF L
    bfe_l = [("493Q", "547T"), ("478T", "493Q"), ("213V", "493Q")]
    # for pst1, pst2 in bfe_l:
    #     log.info("最短路径上的节点 %s-%s", pst1, pst2)
    #     analysis.find_shortest_path(pst1, pst2)

    for pst1, pst2 in bfe_l:
        log.info("邻居节点 %s", pst1)
        analysis.find_neighbour(pst1)
        log.info("邻居节点 %s", pst2)
        analysis.find_neighbour(pst2)
