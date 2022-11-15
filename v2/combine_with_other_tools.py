# 组合数
# 日志
import logging
import os
import re
from itertools import combinations
from typing import List

import matplotlib.pyplot as plt
import networkx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

import logconfig

logconfig.setup_logging()
log = logging.getLogger("cov2")

AA = ['-', 'C', 'F', 'I', 'L', 'M', 'V', 'W', 'Y', 'A', 'T', 'D', 'E', 'G', 'P', 'N', 'Q', 'S', 'H', 'R', 'K']
# origin_aas = ['P681R', 'Q954H', 'Q613H', 'Y449H', 'Y505H', 'G446S', 'N764K', 'R408S', 'A67V', 'S494P', 'G142D', 'N969K',
#               'T547K', 'E516Q', 'E484A', 'N440K', 'K417T', 'V213G', 'Y145H', 'N439K', 'P384L', 'N211I', 'N679K',
#               'D405N', 'L452Q', 'Q677H', 'F490S', 'D796Y', 'A222V', 'L981F', 'T478K', 'T95I', 'V367F', 'N501Y', 'S477N',
#               'N501T', 'T376A', 'H655Y', 'G339D', 'A653V', 'F486V', 'D614G', 'P681H', 'L452X', 'N856K', 'S373P',
#               'K417N', 'S371F', 'S375F', 'S371L', 'Q493R', 'E484K', 'Q498R', 'E484Q', 'R346K', 'A701V', 'L452R',
#               'G496S']
from output_procon_analysis import AnalysisMutationGroup, ProConNetwork

analysisMutationGroup = AnalysisMutationGroup()
origin_aas = analysisMutationGroup.non_duplicated_aas

# sns.set()
sns.set_style("ticks")

data_dir = "../data/procon"


class CombineResult:
    def __init__(self, ):
        self.mutation_groups = AnalysisMutationGroup()
        self.pcn = ProConNetwork(self.mutation_groups, threshold=100)

    def read_procon_single_mutation(self, ):
        pcn = self.pcn
        mutation_group = self.mutation_groups
        type1 = pcn.type1
        nd_aas = mutation_group.non_duplicated_aas

        pst_info = pcn._collect_mutation_info(save=False)
        pst_info = pst_info.reset_index(drop=True)

        aas_info = pd.DataFrame({"name": nd_aas})
        aas_info["aa"] = aas_info["name"].apply(mutation_group.aa2position)
        aas_info = pd.merge(aas_info, pst_info)
        return aas_info

    def analysis_single_mutation(self):
        bfe = self.read_topnettree()
        ddg = self.read_deep_ddg()
        conservation = self.read_procon_single_mutation()
        conservation_columns = conservation.columns.tolist()
        conservation_data_columns = conservation.columns[2:].tolist()

        # single mutation
        info_list = [conservation, bfe, ddg]
        aa_info: pd.DataFrame = pd.concat([i.set_index("name") for i in info_list], axis=1)
        aa_info = aa_info.loc[:, conservation_columns[1:] + ["BFE", "ddG"]]
        aa_info = aa_info.rename(columns={"ddG": "stability", })
        aa_info = aa_info.sort_values("conservation", ascending=True)
        aa_info.to_csv(os.path.join(data_dir, "combine_with_other_tools - single.csv"))

        # relationship
        c1_columns = ["stability", "BFE"]
        c2_columns = conservation_data_columns
        top_names = []
        for i in aa_info.index.values:
            # top_names.append("{}{}".format(i[1:-1], i[0]))
            top_names.append(i)
        correlation = self.cal_correlation(aa_info, c1_columns, c2_columns, "single-info", top_names)
        correlation.to_csv(os.path.join(data_dir, "combine_with_other_tools - single - correlation.csv"))
        return aa_info, correlation

    def read_procon_co_mutation(self, ):
        pcn = self.pcn
        mutation_group = self.mutation_groups
        G: networkx.Graph = pcn.G
        aa_groups = mutation_group.aa_groups
        aa_groups_info = mutation_group.aa_groups_info

        co_info = []
        for i, grp in enumerate(aa_groups):
            name = aa_groups_info.iloc[i]["name"]
            for a1, a2 in combinations(grp, 2):
                p1 = mutation_group.aa2position(a1)
                p2 = mutation_group.aa2position(a2)
                if G.has_edge(p1, p2):
                    coconservation = G.edges[p1, p2]["weight"]
                else:
                    coconservation = None
                path_length = pcn.shortest_path_length[p1][p2]
                co_info.append([name, a1, a2, coconservation, path_length])

        co_info = pd.DataFrame(co_info, columns=["name", "a1", "a2", "co-conservation", "shortest path length"])
        return co_info

    def analysis_co_mutations(self):
        bfe = self.read_topnettree()
        bfe = bfe.set_index("name")
        ddg = self.read_deep_ddg()
        ddg = ddg.set_index("name")
        conservation = self.read_procon_co_mutation()

        def cal_bfe(row):
            a1 = row["a1"]
            a2 = row["a2"]
            if a1 in bfe.index and a2 in bfe.index:
                res = (bfe.loc[a1]["BFE"] + bfe.loc[a2]["BFE"]) / 2
                return res
            return None

        def cal_ddg(row):
            a1 = row["a1"]
            a2 = row["a2"]
            if a1 in ddg.index and a2 in ddg.index:
                res = (ddg.loc[a1]["ddG"] + ddg.loc[a2]["ddG"]) / 2
                return res
            return None

        info = conservation
        info = info.sort_values("co-conservation", ascending=True)
        info["BFE"] = info.apply(cal_bfe, axis=1)
        info["stability"] = info.apply(cal_ddg, axis=1)

        # check null
        data_columns = ["co-conservation", "shortest path length", "BFE", "stability"]
        info = info.drop(["name"], axis=1)
        info = info.drop_duplicates(["a1", "a2"])
        info = info.reset_index(drop=True)

        # relationship
        c1_columns = data_columns[2:]
        c2_columns = data_columns[:2]
        top_names = []
        for i, row in info.iterrows():
            # n = "{}-{}".format(row.a1[1:-1] + row.a1[0], row.a2[1:-1] + row.a2[0])
            n = "{}-{}".format(row.a1, row.a2)
            top_names.append(n)

        correlation = self.cal_correlation(info, c1_columns, c2_columns, "co-info", top_names)
        info.to_csv(os.path.join(data_dir, "combine_with_other_tools - co.csv"))
        correlation.to_csv(os.path.join(data_dir, "combine_with_other_tools - co - correlation.csv"))
        return info, correlation

    def read_topnettree(self, result_file="../tools/try topnettree/data/result 2022年7月20日.txt"):
        with open(result_file) as f:
            lines = f.readlines()
        lines = lines[1:]
        lines = list(map(lambda x: x.strip(), lines))
        lines = np.reshape(lines, (-1, 2)).tolist()
        bfe = []
        for line in lines:
            name = line[0].split()[0]
            score = re.search("\d+\.\d+", line[1])
            score = score.group() if score else None
            bfe.append({"name": name, "score": score})  # print(name, score)

        bfe = pd.DataFrame(bfe)
        bfe["score"] = bfe["score"].astype(float)
        bfe["idx"] = bfe["name"].str[1:-1].astype(int)
        bfe = bfe[bfe["score"].notna()]

        bfe = bfe.sort_values("idx")
        bfe = bfe.reset_index(drop=True)
        bfe = bfe.rename(columns={"score": "BFE"})
        return bfe

    def read_deep_ddg(self, result_file="../tools/try DeepDDG/v2/6vxx.ddg"):
    # def read_deep_ddg(self, result_file="../tools/try DeepDDG/result-QHD43416.ddg"):
        if "result-QHD43416.ddg" in result_file:
            data = pd.read_csv(result_file, delimiter="\s+", skiprows=[0, ], header=None)
            data.columns = "#chain WT ResID Mut ddG".split()
            data["name"] = data.apply(lambda x: "{}{}{}".format(x["WT"], x["ResID"], x["Mut"]), axis=1)
            data.index = data["name"]
            fdata = data.loc[list(filter(lambda x: x[-1] != "X", origin_aas)), :]
            fdata = fdata.sort_values("ResID")
            fdata = fdata.reset_index(drop=True)
            return fdata
        elif "6vxx.ddg" in result_file or "6vyb.ddg" in result_file:
            data = pd.read_csv(result_file, delimiter="\s+", skiprows=[0, ], header=None)
            data.columns = "#chain WT ResID Mut ddG".split()
            data["name"] = data.apply(lambda x: "{}{}{}".format(x["WT"], x["ResID"], x["Mut"]), axis=1)
            data.index = data["name"]
            data = data.sort_values("ResID")
            data = data[data.iloc[:, 0] == "A"]
            fdata = data.loc[list(filter(lambda x: x in data.index.to_list(), origin_aas)), :]
            fdata = fdata.sort_values("ResID")
            return fdata

    def analysis_variant(self, f_single_info=os.path.join(data_dir, "combine_with_other_tools - single.csv"),
                         f_co_info=os.path.join(data_dir, "combine_with_other_tools - co.csv")):
        """
        read ouputs, calculate variant
        """
        # middle info
        single_info = pd.read_csv(f_single_info)
        # variant info
        mutation_group = self.mutation_groups
        aa_groups = mutation_group.aa_groups
        aa_groups_info = mutation_group.aa_groups_info
        group_info = pd.concat(map(pd.Series, aa_groups), keys=aa_groups_info["name"].to_list()).reset_index()
        group_info.columns = ["variant", "num", "name"]
        # single
        variant_single = pd.merge(group_info, single_info)

        # mean
        variant_single_group = variant_single.groupby("variant").mean()
        variant_single_group.to_csv(os.path.join(data_dir, "combine_with_other_tools - variant - single.csv"))

        c1_columns = variant_single_group.columns[-2:].to_list()
        c2_columns = variant_single_group.columns[2:-2].to_list()
        single_correlation = self.cal_correlation(variant_single_group, c1_columns, c2_columns, "variant-single",
                                                  variant_single_group.index.to_list())
        single_correlation.to_csv(
            os.path.join(data_dir, "combine_with_other_tools - variant - single - correlation.csv"))

        # relationship for each variant
        def aux(grp):
            res = self.cal_correlation(grp, c1_columns, c2_columns, "variant-single", grp["name"].to_list())
            return res

        variant_single_variant_correlation = variant_single.groupby("variant").apply(aux)
        variant_single_variant_correlation.to_csv(
            os.path.join(data_dir, "combine_with_other_tools - variant - single - variant - correlation.csv")
        )

        # co
        co_info = pd.read_csv(f_co_info, index_col=0)
        co_data_columns = co_info.columns[-3:].tolist()
        variant_co = []
        for i, grp in enumerate(aa_groups):
            name = aa_groups_info.iloc[i]["name"]
            for a1, a2 in combinations(grp, 2):
                idx = (co_info[["a1", "a2"]] == [a1, a2]) | (co_info[["a1", "a2"]] == [a2, a1])
                idx = idx.all(axis=1)
                if sum(idx) > 0:
                    row = co_info[idx].iloc[0]
                    row["name"] = name
                    variant_co.append(row)
        variant_co = pd.DataFrame(variant_co).reset_index(drop=True)
        # mean
        variant_co_grp = variant_co.groupby("name").mean()
        single_correlation.to_csv(os.path.join(data_dir, "combine_with_other_tools - variant - co.csv"))
        c1_columns = co_data_columns[1:]
        c2_columns = co_data_columns[:1]
        single_correlation = self.cal_correlation(variant_co_grp, c1_columns, c2_columns, "variant-co",
                                                  variant_co_grp.index.to_list())
        single_correlation.to_csv(os.path.join(data_dir, "combine_with_other_tools - variant - co - correlation.csv"))

        # variant correlation
        def aux(grp):
            top_names = []
            for i, row in grp.iterrows():
                # top_names.append("{}-{}".format(row.a1[1:-1] + row.a1[0], row.a2[1:-1] + row.a2[0]))
                top_names.append("{}-{}".format(row.a1, row.a2))
            res = self.cal_correlation(grp, c1_columns, c2_columns, "variant-co", top_names)
            return res

        variant_co_variant_correlation = variant_co.groupby("name").apply(aux)
        variant_co_variant_correlation.to_csv(
            os.path.join(data_dir, "combine_with_other_tools - variant - co - variant - correlation.csv")
        )

    @staticmethod
    def cal_correlation(info: pd.DataFrame, c1_columns: List, c2_columns: List, name: str, top_names=None):
        """
        correlation analysis
        :param info: data source
        :param c1_columns: target1
        :param c2_columns: target2
        :param name: reuslt name
        :return: pd.DataFrame
        """

        def format_top(s: pd.Series, count=3):
            s = s.iloc[s.abs().values.argsort()][::-1][:count]
            psts = []
            print(s)
            for index, value in s.items():
                # position = index[1:-1] + index[0]
                position = "{}({:.2f})".format(index, value)
                psts.append(position)
            return ",".join(psts)

        if top_names is None:
            top_names = info.index.values

        info = info.iloc[:]
        info["top_names"] = top_names
        correlation = []
        for c1 in c1_columns:
            for c2 in c2_columns:
                compare_data = info.loc[:, [c1, c2, "top_names"]]
                # remove empty
                compare_data = compare_data[compare_data.notna().all(axis=1)]
                c1_data = compare_data.loc[:, c1].to_list()
                c2_data = compare_data.loc[:, c2].to_list()
                print("total {}, not empty{}".format(info.shape[0], compare_data.shape[0]))

                if len(c1_data) < 2 or len(c2_data) < 2:
                    p = [None, None]
                    sp = [None, None]
                else:
                    # pearsonr
                    p = pearsonr(c1_data, c2_data)
                    print(f"{c1} / {c2} : pearsonr={p[0]}, p={p[1]}")
                    # spear
                    sp = spearmanr(c1_data, c2_data)
                    print(f"{c1} / {c2} : spearmanr={sp[0]}, p={sp[1]}")
                # top 3
                top_data = pd.Series(compare_data.loc[:, c1].values, index=compare_data.loc[:, "top_names"])
                top3 = format_top(top_data)
                # result
                correlation.append([c1, c2, p[0], p[1], sp[0], sp[1], info.shape[0], compare_data.shape[0], top3])

        correlation = pd.DataFrame(correlation,
                                   columns=["name1", "name2", "pearsonr correlation", "pearsonr p",
                                            "spearmanr correlation", "spearmanr p",
                                            "total data", "not null data", "top3"])
        correlation.name = name
        return correlation

    @staticmethod
    def parse_result(
            f_single_correlation=os.path.join(data_dir, "combine_with_other_tools - single - correlation.csv"),
            f_co_correlation=os.path.join(data_dir, "combine_with_other_tools - co - correlation.csv"),
            f_variant_s_correlation=os.path.join(data_dir,
                                                 "combine_with_other_tools - variant - single - correlation.csv"),
            f_variant_co_correlation=os.path.join(data_dir,
                                                  "combine_with_other_tools - variant - co - correlation.csv")

    ):
        """analysis result files for three types, and draw table and heatmap"""
        names = ["substitution", "co-substitution", "variant (single)", "variant (co)"]
        correlations = [pd.read_csv(i, index_col=[1, 2]) for i in
                        [f_single_correlation, f_co_correlation, f_variant_s_correlation, f_variant_co_correlation]]
        all_cor = pd.concat(correlations, keys=names)

        # format
        all_cor = all_cor.rename(
            {"co-substitution": "substitution", "variant (single)": "variant", "variant (co)": "variant"}, level=0,
            axis=0)
        # all_cor = all_cor.drop(["degree", "average weighted degree", "conservation", "average shortest length in sampled data"], level=2, axis=0)
        # all_cor = all_cor.rename({"weighted degree": "degree", "normalized conservation": "conservation"})

        # count
        # spearmanr is more
        count_p = sum(all_cor["pearsonr p"] < 0.05)
        count_sp = sum(all_cor["spearmanr p"] < 0.05)
        print(f"total {all_cor.shape[0]}, pearson valid {count_p}, spearsanr valid {count_sp}")

        def draw_heatmap(plot_cor):
            # heatmap
            fig: plt.Figure
            axes: List[plt.Axes]
            fig, axes = plt.subplots(1, 2, sharey="row", figsize=(7, 7))
            for i, ax in zip(target_columns, axes[:2]):
                target_data = plot_cor[i].unstack(level=1)
                sns.heatmap(target_data, ax=ax)
                ax.set_xlabel(i)
                ax.set_ylabel("")
            fig.tight_layout()
            # fig.show()
            return fig

        # find data
        target = "spearmanr"
        target_columns = [target + " correlation", target + " p"]
        # total
        plot_cor = all_cor
        fig = draw_heatmap(plot_cor)
        fig.show()
        fig.savefig(os.path.join(data_dir, "combine_with_other_tools - parse - heatmap1.png"))
        # figure with difference
        plot_cor = all_cor.copy()
        idx_invalid = plot_cor[target_columns[-1]] >= 0.05
        plot_cor[target_columns[0]][idx_invalid] = pd.NA
        plot_cor[target_columns[1]][idx_invalid] = pd.NA
        fig = draw_heatmap(plot_cor)
        fig.show()
        fig.savefig(os.path.join(data_dir, "combine_with_other_tools - parse - heatmap2.png"))

        # table with difference
        valid_cor = all_cor.copy()
        valid_cor = valid_cor.unstack(level=1)
        idx_valid = (valid_cor.loc[:, "spearmanr p"] < 0.05).any(axis=1)
        valid_cor = valid_cor[idx_valid]
        valid_cor = valid_cor.stack()  # 变换回去
        valid_cor = valid_cor.swaplevel(0, 2).sort_index(level=0)
        valid_cor = valid_cor.swaplevel(1, 2).sort_index(level=0)
        # format
        valid_cor = valid_cor[target_columns + ["top3"]]
        valid_cor.index.names = map(lambda x: x[0].upper() + x[1:], ["#", "kind", "network characteristic"])
        valid_cor.columns = ["Correlation", "P value", "Top3"]
        # top3
        valid_cor = valid_cor.reset_index()
        valid_cor.insert(2, "Top3", valid_cor.pop("Top3"))
        valid_cor = valid_cor.sort_values(valid_cor.columns.to_list()[:4])
        valid_cor = valid_cor.set_index(valid_cor.columns.to_list()[:4])
        # save
        valid_cor.to_excel(os.path.join(data_dir, "combine_with_other_tools - parse.xlsx"))

    @staticmethod
    def parse_variant_result(
            f_variant_s_correlation=os.path.join(data_dir,
                                                 "combine_with_other_tools - variant - single - variant - correlation.csv"),
            f_variant_co_correlation=os.path.join(data_dir,
                                                  "combine_with_other_tools - variant - co - variant - correlation.csv")
    ):
        correlations = [pd.read_csv(i, index_col=[0, 1, 2, 3]) for i in
                        [f_variant_s_correlation, f_variant_co_correlation]]
        all_cor = pd.concat(correlations)
        all_cor = all_cor.iloc[:, 2:]
        # all_cor[all_cor["not null data"] <= 2][["spearmanr correlation", "spearmanr p"]] = pd.NA
        all_cor = all_cor.droplevel(1, axis=0)
        all_cor = all_cor.swaplevel(0, 1).swaplevel(1, 2).sort_index(axis=0)
        all_cor_sorted = all_cor.reset_index()
        # t1 = all_cor_sorted[(all_cor_sorted.name1=="BFE") & (all_cor_sorted.name2=="co-conservation")]
        # t1_idx = t1["spearmanr correlation"].abs().argsort()
        # t1_idx_2 = t1["spearmanr correlation"].abs().values.argsort()
        # print(t1)
        # print(t1_idx)
        # print(t1.iloc[t1_idx])
        # print(t1_idx_2)
        # print(t1.iloc[t1_idx_2])

        all_cor_sorted = all_cor_sorted.groupby(["name1", "name2"]).apply(
            lambda x: x.iloc[x["spearmanr correlation"].abs().values.argsort()[::-1]])
        all_cor_sorted = all_cor_sorted.droplevel(2)
        all_cor_sorted = all_cor_sorted.iloc[:, 2:]
        all_cor_sorted.to_excel(os.path.join(data_dir, "combine_with_other_tools - parse - variant(all).xlsx"))

        # filter
        all_cor_sorted_valid = all_cor_sorted[all_cor_sorted["spearmanr p"].notna()]
        all_cor_sorted_valid = all_cor_sorted_valid[all_cor_sorted_valid["spearmanr p"] <= 0.05]
        all_cor_sorted_valid = all_cor_sorted_valid[
            all_cor_sorted_valid["spearmanr correlation"].astype(int).abs() != 1]

        # format result
        # all_cor_sorted_valid = all_cor_sorted_valid.drop(["degree", "average weighted degree", "conservation", "average shortest length in sampled data"], level=1,
        #                                                  axis=0)
        # all_cor_sorted_valid = all_cor_sorted_valid.rename(
        #     {"weighted degree": "degree", "normalized conservation": "conservation", "average shortest length in variant": "average shortest length"})
        all_cor_sorted_valid.index.names = ["Kind", "Network characteristic"]
        all_cor_sorted_valid.columns = map(lambda x: x[0].upper() + x[1:], all_cor_sorted_valid.columns)

        all_cor_sorted_valid = all_cor_sorted_valid.reset_index()
        all_cor_sorted_valid.insert(0, "Variant", all_cor_sorted_valid.pop("Variant"))
        all_cor_sorted_valid.insert(2, "Top3", all_cor_sorted_valid.pop("Top3"))
        all_cor_sorted_valid = all_cor_sorted_valid.sort_values("Variant")
        all_cor_sorted_valid = all_cor_sorted_valid.set_index(all_cor_sorted_valid.columns.to_list()[:4])
        # format
        all_cor_sorted_valid = all_cor_sorted_valid.iloc[:, :-2]
        all_cor_sorted_valid = all_cor_sorted_valid.rename(
            {"Spearmanr correlation": "Correlation", "Spearmanr p": "P value"}, axis=1)

        # save
        all_cor_sorted_valid.to_excel(os.path.join(data_dir, "combine_with_other_tools - parse - variant.xlsx"))

    @staticmethod
    def plot_correlation_scatter(
            single_info_path="../data/procon/combine_with_other_tools - single.csv",
            co_info_path="../data/procon/combine_with_other_tools - co.csv"
    ):
        # read data
        single_info: pd.DataFrame = pd.read_csv(single_info_path, index_col=0)
        co_info: pd.DataFrame = pd.read_csv(co_info_path, index_col=0)

        # draw
        plot_data = {
            "BFE": [
                {"type": "co", "name": "co-conservation"},
                {"type": "co", "name": "shortest path length"},
            ],
            "stability": [
                {"type": "single", "name": "degree"},
                {"type": "single", "name": "page rank"},
            ]
        }
        abbr_map = {
            "co-conservation": "CCS",
            "shortest path length": "L",
            "degree": "Kw",
            "page rank": "P"
        }

        for name, targets in plot_data.items():
            for idx, target in enumerate(targets):
                fig: plt.Figure
                fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
                source = co_info if target["type"] == "co" else single_info
                sns.regplot(data=source, x=name, y=target["name"], ax=ax)
                ax.set_ylabel(abbr_map[target["name"]])
                fig.savefig(os.path.join(data_dir, f"correlation_{name} {idx}.png"))

    @staticmethod
    def get_graph_top3(
            single_info_path="../data/procon/combine_with_other_tools - single.csv",
            co_info_path="../data/procon/combine_with_other_tools - co.csv"
    ):
        # co
        co_info: pd.DataFrame = pd.read_csv(co_info_path, index_col=0)
        co_info["a1"] = co_info["a1"].apply(lambda x: x[1:-1] + x[0])
        co_info["a2"] = co_info["a2"].apply(lambda x: x[1:-1] + x[0])
        co_target_names = ["co-conservation", "shortest path length"]
        co_res = []
        for n in co_target_names:
            co_info_sorted = co_info.sort_values(n, ascending=False)
            top3 = co_info_sorted.iloc[:3]
            top3 = top3.apply(lambda x: f"{x.a1}-{x.a2}", axis=1).to_list()
            top3 = ",".join(top3)
            co_res.append({"Network characteristic": n, "Top3": top3})

        # single
        single_info: pd.DataFrame = pd.read_csv(single_info_path)
        single_target_names = single_info.columns.tolist()[2:-2]
        single_res = []
        for n in single_target_names:
            single_info_sorted = single_info.sort_values(n, ascending=False)
            top3 = single_info_sorted[:3]
            top3 = ",".join(top3["aa"].to_list())
            single_res.append(
                {"Network characteristic": n, "Top3": top3}
            )
        top3_res = single_res + co_res
        top3_res = pd.DataFrame(top3_res)
        top3_res.index = top3_res.index.values + 1
        top3_res.index.name = "#"
        top3_res.to_excel(os.path.join(data_dir, "network characteristic top3.xlsx"))

    @staticmethod
    def plot_for_stability_bfe(single_info_path="../data/procon/combine_with_other_tools - single.csv"):
        single_info = pd.read_csv(single_info_path, index_col=None)
        plot_data = single_info[["BFE", "stability"]]
        plot_data = plot_data[plot_data.notna().all(axis=1)]
        fig, axes = plt.subplots(constrained_layout=True, figsize=(6.4, 4.8))
        sns.histplot(data=plot_data, x="BFE", stat="probability", bins=10, ax=axes, )
        fig.show()
        fig.savefig(os.path.join(data_dir, "BFE distribution.png"), dpi=300)

        fig, axes = plt.subplots(constrained_layout=True, figsize=(6.4, 4.8))
        sns.histplot(data=plot_data, x="stability", stat="probability", bins=10, ax=axes, )
        fig.show()
        fig.savefig(os.path.join(data_dir, "stability distribution.png"), dpi=300)
        # print(plot_data)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    # analysis
    mutation_groups = AnalysisMutationGroup()
    pcn = ProConNetwork(mutation_groups, threshold=100)
    cr = CombineResult()

    # single mutation
    singe_info = cr.analysis_single_mutation()
    # co mutation
    co_info = cr.analysis_co_mutations()
    # variant
    cr.analysis_variant()

    # parse result
    CombineResult.parse_result()
    CombineResult.parse_variant_result()

    # draw scatter
    CombineResult.plot_correlation_scatter()

    # top3
    CombineResult.get_graph_top3()

    # draw distribution
    CombineResult.plot_for_stability_bfe()
