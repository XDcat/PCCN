import json
import math
import seaborn as sns
from Bio import SeqIO
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 日志
import logging
import logconfig

logconfig.setup_logging()
log = logging.getLogger("cov2")


class ProConNetwork:
    def __init__(self,
                 analysis="../data/procon/analysis.json",
                 parse1="../data/procon/type1_parse.csv",
                 parse2="../data/procon/type2_parse.csv",
                 fasta_file="../data/YP_009724390.1.txt",
                 procon_threshold=300,
                 ):
        self.procon_threhold = procon_threshold  # 确定网络阈值
        # 关注的变异组的相关数据
        with open(analysis) as f:
            self.analysis = json.load(f)

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

    def _get_nodes(self, fasta, type1):
        nodes = ["{}{}".format(i + 1, j) for i, j in enumerate(fasta)]
        nodes = pd.DataFrame({"position": nodes})
        nodes = pd.merge(nodes, type1, how="left", left_on="position", right_on="position")
        nodes = nodes.loc[:, ["position", "information"]]
        nodes.columns = ["name", "symbolSize"]
        nodes = nodes.fillna(0)
        log.debug("nodes = %s", nodes)
        return nodes

    def _get_links(self, type2, ):
        type2 = type2[type2["info"] > self.procon_threhold]  # TODO: 选择阈值
        log.debug("type2 = %s", type2)
        links = type2.loc[:, ["site1", "site2"]]
        links.columns = ["source", "target"]
        log.debug("links = %s", links)
        return links

    def _get_G(self, links, nodes):
        # 绘制关系图
        G = nx.Graph()
        for i, row in nodes.iterrows():
            G.add_node(row["name"], name=nodes["name"], )

        for i, row in links.iterrows():
            G.add_edge(row["source"], row["target"])
        return G

    def display(self):
        log.info("self.nodes = %s", self.nodes)
        log.info("self.links = %s", self.links)
        log.info("self.G = %s", self.G)

if __name__ == '__main__':
    pcn = ProConNetwork()
    pcn.display()