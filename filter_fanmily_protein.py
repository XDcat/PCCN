# -*- coding:utf-8 -*-
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import json
import logging
import logconfig

logconfig.setup_logging()
log = logging.getLogger("cov2")


def load_fasta(fasta_file):
    seqs = SeqIO.parse(fasta_file, "fasta")
    seqs = [[i.id.split("|")[0], i.id, i.seq] for i in seqs]
    columns = ["id", "desc", "seq", ]
    seqs = pd.DataFrame(seqs, columns=columns)
    return seqs


def load_json(json_file):
    with open(json_file) as f:
        info = json.load(f)
    tinfo = []
    for i in info:
        metadata = i["metadata"]
        metadata.update(metadata.pop("source_organism"))
        tinfo.append(metadata)
    info = pd.DataFrame(tinfo)
    # info = info.rename({"accession": "id"})
    return info


if __name__ == '__main__':
    fasta_file = "./data/protein-matching-IPR042578.fasta"
    json_file = "./data/protein-matching-IPR042578.json"
    query_fasta_file = "./data/YP_009724390.1.txt"
    fasta = load_fasta(fasta_file)
    info = load_json(json_file)
    log.debug("fasta columns: %s", fasta.columns)
    log.debug("fasta shape %s ", fasta.shape)
    log.debug("info columns: %s", info.columns)
    log.debug("info shape %s ", info.shape)
    # filter
    log.debug("info no duplicated: %s", info.shape[0] - info.fullName.duplicated().sum())
    data = pd.merge(fasta, info, left_on="id", right_on="accession")
    log.debug("merge:\n%s", data)
    # data = data[data.fullName.duplicated()]
    data = data.drop_duplicates("fullName")
    log.debug("drop duplicates:\n%s", data)

    # output csv and fasta
    data.to_csv("./data/protein-matching-IPR042578.filter.csv")
    res_fasta = list(SeqIO.parse(query_fasta_file, "fasta"))
    for i, row in data.iterrows():
        rec = SeqRecord(
            Seq(row.seq),
            id=row.id,
            # description="|".join([row.fullName, row.source_database])
            description="| " + " | ".join([row.source_database, row.fullName, ])
        )
        res_fasta.append(rec)
    SeqIO.write(res_fasta, "./data/protein-matching-IPR042578.filter.fasta", "fasta")
