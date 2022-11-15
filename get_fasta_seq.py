# -*- coding:utf-8 -*-
import pandas as pd
import requests
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FILE_PATH = r"./data/YP_009724390.1.25.tsv"
NAMES = "qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore".split()

def get_genes():
    data = pd.read_csv(FILE_PATH, delimiter="\s", header=None, names=NAMES)
    # filter
    data = data[data.qseqid != data.sseqid]
    logger.debug(data)
    return data["sseqid"]


def craw(gene, to_path):
    """get gene content"""
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
            logger.warning("{} error".format(gene))
    except Exception as e:
        print(e)




if __name__ == '__main__':
    to_path = "./data/YP_009724390.1.25.fasta"
    craw("YP_009724390.1", to_path)
    genes = get_genes()
    for g in genes:
        craw(g, to_path)