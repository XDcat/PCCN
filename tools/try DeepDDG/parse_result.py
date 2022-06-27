import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
origin_aas = ['P681R', 'Q954H', 'Q613H', 'Y449H', 'Y505H', 'G446S', 'N764K', 'R408S', 'A67V', 'S494P', 'G142D', 'N969K',
              'T547K', 'E516Q', 'E484A', 'N440K', 'K417T', 'V213G', 'Y145H', 'N439K', 'P384L', 'N211I', 'N679K',
              'D405N', 'L452Q', 'Q677H', 'F490S', 'D796Y', 'A222V', 'L981F', 'T478K', 'T95I', 'V367F', 'N501Y', 'S477N',
              'N501T', 'T376A', 'H655Y', 'G339D', 'A653V', 'F486V', 'D614G', 'P681H', 'L452X', 'N856K', 'S373P',
              'K417N', 'S371F', 'S375F', 'S371L', 'Q493R', 'E484K', 'Q498R', 'E484Q', 'R346K', 'A701V', 'L452R',
              'G496S']
def read_deep_ddg( result_file="./result-QHD43416.ddg" ):
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
    print(fdata)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=fdata, x="name", y="ddG", ax=ax, color="C0")
    [i.set_rotation(90) for i in ax.get_xticklabels()]
    ax.set_xlabel("")
    ax.set_ylabel("Stability changes (kcal/mol)")
    fig.tight_layout()
    fig.show()
    fig.savefig("barplot.png")
