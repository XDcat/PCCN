import pandas as pd
from itertools import combinations

origin_aas = ['P681R', 'Q954H', 'Q613H', 'Y449H', 'Y505H', 'G446S', 'N764K', 'R408S', 'A67V', 'S494P', 'G142D', 'N969K',
              'T547K', 'E516Q', 'E484A', 'N440K', 'K417T', 'V213G', 'Y145H', 'N439K', 'P384L', 'N211I', 'N679K',
              'D405N', 'L452Q', 'Q677H', 'F490S', 'D796Y', 'A222V', 'L981F', 'T478K', 'T95I', 'V367F', 'N501Y', 'S477N',
              'N501T', 'T376A', 'H655Y', 'G339D', 'A653V', 'F486V', 'D614G', 'P681H', 'L452X', 'N856K', 'S373P',
              'K417N', 'S371F', 'S375F', 'S371L', 'Q493R', 'E484K', 'Q498R', 'E484Q', 'R346K', 'A701V', 'L452R',
              'G496S']

if __name__ == '__main__':
    old_result = "./v4/output/result total v1.csv"
    old_data = pd.read_csv(old_result)
    all_pairwise = list(combinations(origin_aas, 2))
    # print(all_pairwise)

    done_pairwise = []
    for i in old_data["mutations"]:
        i = i.split(";")
        a1 = i[0][2:]
        a2 = i[1][2:]
        done_pairwise.append((a1, a2))

    wait_pairwise = []
    for a1, a2 in all_pairwise:
        if "X" in a1 or "X" in a2:
            continue
        if (a1, a2) in done_pairwise:
            continue
        if (a2, a1) in done_pairwise:
            continue
        else:
            wait_pairwise.append((a1, a2))

    print(f"aa 个数:{len(origin_aas)}")
    print(f"pairwise: all done wait")
    print(f"pairwise: {len(all_pairwise)} {len(done_pairwise)} {len(wait_pairwise)}")


    sep = 400
    for i in range(len(wait_pairwise) // sep + 1):
        out_file = "{}-{}.txt".format(i * sep, (i + 1) * sep)
        print(out_file)
        with open(out_file, "w") as f:
            for a1, a2 in wait_pairwise[i * sep: (i + 1) * sep]:
                f.write("A {};A {}\n".format(a1, a2))


