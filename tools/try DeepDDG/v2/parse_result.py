import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# sns.set()
sns.set_style("ticks")  # 主题: 白色背景且有边框
# 更新字体大小
plt.rcParams.update({'font.size': 16})
plt.rcParams["axes.titlesize"] = "medium"
aas = ['E516Q', 'T1027I', 'S982A', 'P681R', 'L452R', 'A222V', 'R190S', 'L24S', 'T716I', 'P9L', 'L5F', 'D1118H', 'Q52R',
       'Q613H', 'T19R', 'E484Q', 'E484A', 'D138Y', 'K417T', 'Q1071H', 'S13I', 'Q954H', 'F486V', 'F888L', 'K417N',
       'W152C', 'V213G', 'T376A', 'D80A', 'Q493R', 'L18F', 'H245Y', 'Y145N', 'N969K', 'L981F', 'E484K', 'S375F',
       'C136F',
       'G75V', 'H655Y', 'T478K', 'N211I', 'N439K', 'G339D', 'D80G', 'S477N', 'Q498R', 'N679K', 'G142D', 'N856K',
       'I1130V', 'T76I', 'A67V', 'T95I', 'V367F', 'E780K', 'T20N', 'V1176F', 'Y145H', 'G1219V', 'D1139H', 'R408S',
       'S373P', 'R246N', 'A570D', 'D796Y', 'R346K', 'D950N', 'H245P', 'Y449H', 'T19I', 'P26S', 'D796H', 'N440K',
       'P384L',
       'V126A', 'D215G', 'Q677H', 'F157L', 'L452Q', 'D614G', 'F490S', 'N501T', 'T547K', 'T732A', 'H1101Y', 'Y505H',
       'D405N', 'E1092K', 'A701V', 'N764K', 'E156G', 'S371L', 'D253G', 'A653V', 'S371F', 'P681H', 'N501Y', 'T859N',
       'S494P', 'G496S']
aas = sorted(aas, key=lambda x: int(x[1:-1]))


def read_deep_ddg(result_file="./6vyb.ddg"):
    data = pd.read_csv(result_file, delimiter="\s+", skiprows=[0, ], header=None)
    data.columns = "#chain WT ResID Mut ddG".split()
    data["name"] = data.apply(lambda x: "{}{}{}".format(x["WT"], x["ResID"], x["Mut"]), axis=1)
    data.index = data["name"]
    data = data.sort_values("ResID")
    data = data[data.iloc[:, 0] == "A"]
    fdata = data.loc[list(filter(lambda x: x in data.index.to_list(), aas)), :]
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
