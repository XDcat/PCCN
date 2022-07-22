from Bio import SeqIO

# v1
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

if __name__ == '__main__':
    print(len(origin_aas))
    # 序列
    origin_fasta = SeqIO.read("../../data/YP_009724390.1.txt", "fasta")
    pdb_a_fasta = SeqIO.read("./data/7A98_A.fasta", "fasta")
    print(f"origin: \n{origin_fasta.seq}")
    print(f"pdf a:\n{pdb_a_fasta.seq}")
    start_sep_count = 31  # origin 序列和 pdb_a 对齐后，pdb_a 前比 origin 多了 31 个

    # 生成新的替代
    run_format = "sh run_topnettree.sh 7a98.pdb A {w} {m} {idx} 1 >> out0.log 2>&1"
    pdb_a_aas = []
    run_sh = []
    for aa in origin_aas:
        w, idx, m = aa[0], int(aa[1:-1]), aa[-1]  # wild, index, mutation
        idx_to_pda_a = idx + 31
        # 校验是否准确
        flag = pdb_a_fasta.seq[idx_to_pda_a - 1] == w
        new_aa = "{}{}{}".format(w, idx_to_pda_a, m)
        print(aa, origin_fasta[idx - 1], pdb_a_fasta[idx_to_pda_a - 1], flag, "->", new_aa)
        if not flag:
            raise RuntimeError("序列不匹配")
        pdb_a_aas.append(new_aa)
        run_sh.append(run_format.format(w=w, idx=idx_to_pda_a, m=m))

    with open("./data/aas.txt", "w") as f:
        for i, j, k in zip(origin_aas, pdb_a_aas, run_sh):
            f.write(f"{i} {j} :{k}\n")
