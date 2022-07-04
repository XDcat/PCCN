import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set()
sns.set_style("ticks")  # 主题: 白色背景且有边框
# 更新字体大小
plt.rcParams.update({'font.size': 16})
plt.rcParams["axes.titlesize"] = "medium"

def read_result(result_file="./data/result.txt"):
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
        bfe.append({"name": name, "score": score})
        # print(name, score)

    bfe = pd.DataFrame(bfe)
    bfe["score"] = bfe["score"].astype(float)
    bfe["idx"] = bfe["name"].str[1:-1].astype(int)
    bfe = bfe[bfe["score"].notna()]

    bfe = bfe.sort_values("idx")

    return bfe


if __name__ == '__main__':
    bfe = read_result()
    print(bfe.dtypes)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(14, 4.8))
    sns.barplot(data=bfe, x="name", y="score", ax=ax, color="C0")
    [i.set_rotation(90) for i in ax.get_xticklabels()]
    ax.set_xlabel("")
    ax.set_ylabel("BFE changes (kcal/mol)")
    fig.tight_layout()
    fig.show()
    fig.savefig("./data/barplot.png")

    print(bfe)
    bfe.to_csv("./data/result_precess.csv")
