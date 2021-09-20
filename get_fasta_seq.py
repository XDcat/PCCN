# -*- coding:utf-8 -*-
'''
__author__ = 'XD'
__mtime__ = 2019/12/11
__project__ = Ponsol
Fix the Problem, Not the Blame.

计算 diamond 找到的同原序列的序列
'''
import pandas as pd
import requests
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FILE_PATH = r"./data/YP_009724390.1.25.tsv"
NAMES = "qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore".split()

def get_genes():
    data = pd.read_csv(FILE_PATH, delimiter="\s", header=None, names=NAMES)
    # 筛除自身
    data = data[data.qseqid != data.sseqid]
    logger.debug(data)
    return data["sseqid"]


def craw(gene, to_path):
    """ 爬取基因内容 """
    url = "https://www.ncbi.nlm.nih.gov/sviewer/viewer.fcgi?id={}&db=protein&report=fasta&retmode=text&withmarkup=on&tool=portal&log$=seqview&maxdownloadsize=1000000"
    url = url.format(gene)
    try:
        logger.debug("开始爬取{}".format(gene))
        logger.debug(url)
        response = requests.get(url)
        logger.info(response.status_code)
        if response.status_code == 200:
            with open(to_path, "a") as f:
                f.write(response.text.strip() + "\n")
        else:
            logger.warning("{}出错".format(gene))
    except Exception as e:
        print(e)




if __name__ == '__main__':
    to_path = "./data/YP_009724390.1.25.fasta"
    craw("YP_009724390.1", to_path)
    genes = get_genes()  # 获取所有基因
    for g in genes:
        craw(g, to_path)