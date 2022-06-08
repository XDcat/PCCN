from Bio import SeqIO
origin_aas = ['P681R', 'Q954H', 'Q613H', 'Y449H', 'Y505H', 'G446S', 'N764K', 'R408S', 'A67V', 'S494P', 'G142D', 'N969K',
              'T547K', 'E516Q', 'E484A', 'N440K', 'K417T', 'V213G', 'Y145H', 'N439K', 'P384L', 'N211I', 'N679K',
              'D405N', 'L452Q', 'Q677H', 'F490S', 'D796Y', 'A222V', 'L981F', 'T478K', 'T95I', 'V367F', 'N501Y', 'S477N',
              'N501T', 'T376A', 'H655Y', 'G339D', 'A653V', 'F486V', 'D614G', 'P681H', 'L452X', 'N856K', 'S373P',
              'K417N', 'S371F', 'S375F', 'S371L', 'Q493R', 'E484K', 'Q498R', 'E484Q', 'R346K', 'A701V', 'L452R',
              'G496S']


if __name__ == '__main__':
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
        w, idx, m = aa[0], int(aa[1:-1]), aa[-1]  #  wild, index, mutation
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





