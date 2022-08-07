import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# sns.set()
sns.set_style("ticks")  # 主题: 白色背景且有边框
# 更新字体大小
plt.rcParams.update({'font.size': 16})
plt.rcParams["axes.titlesize"] = "medium"
# origin_aas = ['P681R', 'Q954H', 'Q613H', 'Y449H', 'Y505H', 'G446S', 'N764K', 'R408S', 'A67V', 'S494P', 'G142D', 'N969K',
#               'T547K', 'E516Q', 'E484A', 'N440K', 'K417T', 'V213G', 'Y145H', 'N439K', 'P384L', 'N211I', 'N679K',
#               'D405N', 'L452Q', 'Q677H', 'F490S', 'D796Y', 'A222V', 'L981F', 'T478K', 'T95I', 'V367F', 'N501Y', 'S477N',
#               'N501T', 'T376A', 'H655Y', 'G339D', 'A653V', 'F486V', 'D614G', 'P681H', 'L452X', 'N856K', 'S373P',
#               'K417N', 'S371F', 'S375F', 'S371L', 'Q493R', 'E484K', 'Q498R', 'E484Q', 'R346K', 'A701V', 'L452R',
#               'G496S']
origin_aas = ['F486V', 'A222V', 'F888L', 'Q954H', 'R246N', 'F490S', 'E484Q', 'N764K', 'V367F', 'T95I', 'T19I', 'S982A',
              'D796H', 'G339D', 'R190S', 'L24S', 'N501Y', 'D405N', 'S373P', 'E156G', 'E1092K', 'P681H', 'Q52R',
              'H1101Y', 'D796Y', 'D215G', 'T478K', 'Q613H', 'P681R', 'N439K', 'S477N', 'D950N', 'G142D', 'A701V',
              'T732A', 'T20N', 'H245Y', 'A570D', 'G75V', 'L981F', 'R346K', 'V1176F', 'N211I', 'Q1071H', 'P26S', 'K417T',
              'Y449H', 'G1219V', 'P9L', 'N969K', 'L5F', 'S371F', 'A67V', 'E780K', 'L452Q', 'N501T', 'N679K', 'T716I',
              'T376A', 'S13I', 'D253G', 'H655Y', 'K417N', 'D1118H', 'D80G', 'W152C', 'N440K', 'L452R', 'F157L', 'Y505H',
              'P384L', 'S371L', 'D614G', 'C136F', 'H245P', 'Q677H', 'D1139H', 'T859N', 'Y145N', 'Q498R', 'R408S',
              'S494P', 'I1130V', 'D80A', 'G496S', 'S375F', 'V213G', 'E516Q', 'T19R', 'L18F', 'N856K', 'A653V', 'Y145H',
              'E484K', 'T76I', 'V126A', 'T1027I', 'T547K', 'D138Y', 'E484A', 'Q493R']


def read_deep_ddg(result_file="./result-QHD43416.ddg"):
    data = pd.read_csv(result_file, delimiter="\s+", skiprows=[0, ], header=None)
    data.columns = "#chain WT ResID Mut ddG".split()
    data["name"] = data.apply(lambda x: "{}{}{}".format(x["WT"], x["ResID"], x["Mut"]), axis=1)
    data.index = data["name"]
    # for i in origin_aas:
    #     if i not in data["name"]:
    #         print(i)
    # print(data)
    # print(data[data["ResID"] == 452])
    # print(data.columns)
    # print(data.dtypes)
    fdata = data.loc[list(filter(lambda x: x[-1] != "X", origin_aas)), :]
    fdata = fdata.sort_values("ResID")
    return fdata


if __name__ == '__main__':
    fdata = read_deep_ddg()
    fdata = fdata.sort_values("ddG")
    print(fdata)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(4.8, 24))
    sns.barplot(data=fdata, y="name", x="ddG", ax=ax, color="C0")
    # [i.set_rotation(90) for i in ax.get_xticklabels()]
    ax.set_ylabel("")
    ax.set_xlabel("Stability changes (kcal/mol)")
    fig.tight_layout()
    fig.show()
    fig.savefig("Supplemental Figure 10. Stability of substitutions.png")
