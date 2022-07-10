import json

import pandas as pd
import requests
from retry import retry
from Bio import SeqIO


def read_data(data_path="./data/总结 - 20220709.xlsx"):
    """读取所有毒株"""
    data = pd.read_excel(data_path, sheet_name=1)
    variants = data["Lineage + additional mutations"].to_list()
    return variants


@retry(tries=3, delay=10)
def get_mutations(variant: str):
    """需要区分类似B.1.617.2 + E484X (d) 的情况"""
    info_url = "https://outbreak.info/situation-reports?pango={}"
    data_url = "https://api.outbreak.info/genomics/lineage-mutations?pangolin_lineage={}&frequency=0.75"
    if "+" in variant:
        origin_variant = variant
        variant = origin_variant.split("+")[0].strip()
        other_mutation = origin_variant.split("+")[1].strip()
    headers = header()
    session = requests.session()
    print(f"load {data_url.format(variant)}")
    response = session.get(data_url.format(variant), headers=headers)
    data = json.loads(response.text)
    result = data["results"].get(variant, "")
    return result


def run():
    data = read_data()
    print("读取到的毒株:", data)
    mutations = {}
    for variant in data:
        print("获取突变:", variant)
        mutations[variant] = get_mutations(variant)
    with open("./data/mutation_info.json", "w") as f:
        f.write(json.dumps(mutations))
    print(mutations)


def header():
    h = {
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "authorization": "Bearer 0ed52bbfb6c79d1fd8e9c6f267f9b6311c885a4c4c6f037d6ab7b3a40d586ad0",
        "origin": "https://outbreak.info",
        "referer": "https://outbreak.info/",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "Windows",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
    }
    return h


def check_mutation(mutation):
    # read seq
    seq = SeqIO.read("./data/YP_009724390.1.txt", "fasta")
    seq = seq.seq

    # parse mutation
    mutation = mutation.upper()
    ref, pst, alt = mutation[0], int(mutation[1:-1]), mutation[-1]

    # check
    if seq[pst - 1] == ref:
        return True
    else:
        print(f"warning: mutation is {mutation}, but ref aa should be {seq[pst - 1]}")
        return False

def parse_result():
    variants = read_data()
    with open("./data/mutation_info.json", ) as f:
        data = json.load(f)
    # 只保留 S 上的替换
    for k, v in data.items():
        data[k] = list(filter(lambda x: (x["gene"] == "S") and (x["type"] == "substitution"), data[k]))  # S 上的替换
        data[k] = [i["mutation"][2:].upper() for i in data[k]]
        data[k] = list(sorted(data[k], key=lambda x: int(x[1:-1])))
        # 添加补充的
        if "+" in k:
            other_mutation = k.split("+")[1].strip()
            data[k].append(other_mutation)
        data[k] = list(filter(lambda x: "X" not in x, data[k]))

        check_result = list(map(check_mutation, data[k]))
        assert sum(check_result) == len(check_result)
        print(k, ", ".join(data[k]))
    print(data)
    result = {k: " ".join(v) for k, v in data.items()}
    result = pd.Series(result)
    print(result)

    result.to_excel("./data/mutation_info.xlsx")


if __name__ == '__main__':
    # run()  # 爬虫获取突变
    parse_result()  # 解析结果
