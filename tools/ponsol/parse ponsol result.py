# -*- coding:utf-8 -*-
'''
__author__ = 'XD'
__mtime__ = 2021/12/29
__project__ = Cov2_protein
Fix the Problem, Not the Blame.
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 初始化参数
sns.set()

if __name__ == '__main__':
    # result_path = "./solubility all result.txt"
    # data = pd.read_csv(result_path, sep="\t")
    # data = data.applymap(lambda x: "0" if x == "-" else x, )
    # data.iloc[:, 2:] = data.iloc[:, 2:].astype(int)
    # # data.to_excel(result_path[:-3] + "xlsx")
    # print(data)
    # sns.heatmap(data.iloc[:, 2:])
    # plt.show()
    #
    # value_data = data.iloc[:, 2:].values
    # value_data_flatten = [j for i in value_data for j in i]
    # value_count = pd.value_counts(value_data_flatten)
    # print(value_data_flatten)
    # print(value_count)
    # value_count.plot(kind="pie")
    # plt.show()

    solubility_count = pd.DataFrame({
        "all mutations": [11570, 12494, 123],
        "important mutations": [9, 48, 1]
    }, index=["decrease", "no effect", "increase"])
    print(solubility_count)
    # fig:plt.Figure = plt.figure(figsize=(20, 5))
    ax1, ax2 = solubility_count.plot(kind="pie", subplots=True, figsize=(10, 5))
    # plt.pie(solubility_count.iloc[:, 0], labels=solubility_count.index)
    ax1.set_xlabel(solubility_count.columns[0])
    ax2.set_xlabel(solubility_count.columns[1])
    ax1.set_ylabel("")
    ax2.set_ylabel("")
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    fig: plt.Figure = plt.gcf()
    ax1_label = ax1.get_label()
    # fig.legend(handles=[ax1, ], loc="upper left")
    # ax1.legend(loc="upper left")
    # ax2.set_ytitle("")
    fig.tight_layout()
    print([ax1, ], ax1.get_label)
    # print(fig.get_label())
    fig.show()
    fig.savefig("distribution.png", dpi=300)

