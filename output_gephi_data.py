import json
import math
import os

from Bio import SeqIO

import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline, interp1d
from pyecharts import options as opts
from pyecharts.charts import Graph
import logging
import logconfig

import matplotlib.pyplot as plt



class AnalysisMutationGroup:
    def __init__(self, analysis="./data/procon/analysis.json"):
        # find important variants
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


if __name__ == '__main__':
    # find no duplicated mutations
    aas = AnalysisMutationGroup().get_non_duplicated_aas()
    aas = list(set([i[1:-1] + i[0] for i in aas]))
    # path
    type1_file = "./data/procon/type1_parse.csv"
    type1_file_out = "./tools/gephi/type1_parse.csv"
    type2_file = "./data/procon/type2_parse.csv"
    type2_file_out = "./tools/gephi/type2_parse.csv"

    type1: pd.DataFrame = pd.read_csv(type1_file)
    type1 = type1.rename(columns={"position": "ID"})
    type1["is_mutation"] = type1["ID"].apply(lambda x: x in aas)
    type1.to_csv(type1_file_out, index=False)

    type2: pd.DataFrame = pd.read_csv(type2_file)
    type2 = type2.rename(columns={"site1": "Source", "site2": "Target"})
    type2.to_csv(type2_file_out, index=False)
