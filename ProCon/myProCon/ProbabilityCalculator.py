# -*- coding: utf-8 -*-
# @Time : 2019/8/9 17:28
# @Author : Edgar Qian
# @Email : qianneng_se@163.com
# @File : ProbabilityCalculator.py

import math
from myProCon import RankProbaa


class ProbabilityCalculator:
    # Initialization of capitalization for 20 amino acids and 1 vacancy site
    residue = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']

    # Initialization of the full name of 20 amino acids and 1 vacancy site
    full_residue = ["Alanine", "Cysteine", "Aspartic acid", "Glutamic acid", "Phenylalanine", "Glycine", "Histidine",
                    "Isoleucine", "Lysine", "Leucine", "Methionine", "Asparagine", "Proline", "Glutamine", "Arginine",
                    "Serine", "Threonine", "Valine", "Tryptophan", "Tyrosine", "gap"]

    # Initialize the three-position abbreviations of 20 amino acids without empty sites
    tri_residue = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO",
                   "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]

    # An abbreviation for the amino acid in initialization 20, with the empty site marked "-"
    residue_small = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'y',
                     '-']

    aa_pc = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']

    __fileFASTA = ''
    __available = True
    __sequence_number = 0
    __residus_number = 0
    LOG20 = 2.99573
    __rpaa = []  # RankProaa is included in the list
    # (the name of the amino acid + the probability of each amino acid appearing at each site).
    # Amino acids are divided into 20 + 1 groups.
    __freqaa = []
    __gap_20 = []
    __gap_6 = []
    __prob6aa = []
    __freq6aa = []  # A two-dimensional list (the probability of each amino acid appearing at each site)
    # in which the amino acids are divided into 6 + 1 groups
    __entropy_0 = []
    __entropy_1 = []
    __seq_tmp = []
    __seq = []
    __name_array = []
    __reference = 0
    __gap_percent = 0  # The probability threshold of gap site
    # (when the probability of gap occurring at a locus exceeds the threshold,
    # the site is not included in the calculation of conservative site)
    __pvalue_1 = 0  # 显著水平
    __pvalue_2 = 0
    __number_of_group = 0
    __group = []
    __default_group = True
    __group_names = []

    locator = ['-', 'C', 'F', 'I', 'L', 'M', 'V', 'W', 'Y', 'A', 'T', 'D', 'E', 'G', 'P', 'N', 'Q', 'S', 'H', 'R', 'K']

    def __init__(self, file, gap_percent=0.1, p_1=0.01, p_2=0.05):
        self.__gap_percent = gap_percent
        self.__pvalue_1 = p_1
        self.__pvalue_2 = p_2
        self.__reference = 0
        self.__available = True
        self.__default_group = True
        self.__number_of_group = 6
        self.__group = [1, 0, 2, 2, 0, 3, 5, 0, 5, 0, 0, 4, 3, 4, 5, 4, 1, 0, 0, 0, 6]
        self.__group_names = ["hydrophobic", "AT", "negative", "conformational", "polar", "positive", "gap"]
        self.__fileFASTA = file
        self.__residus_number = self.calculate_res_number()
        print("residusNumber = {}".format(self.__residus_number))
        if self.__residus_number <= 0:
            self.__available = False
        # # Initialization probability sequence (when amino acids are divided into 20 + 1 group and 6 + 1 group)
        try:
            for i in range(self.__residus_number):
                self.__rpaa.append([])  # Create a list of amino acid names and occurrence probabilities
                # for each site in the amino acid sequence (20 + 1 group)
                self.__prob6aa.append([])  # Create a list of amino acid occurrences
                # for each site in the amino acid sequence (6 + 1 group)
            for i in range(self.__residus_number):
                for j in range(21):
                    # Initialization of amino acid names and occurrence probabilities
                    # on the site dictionary (initial values 0.0, 20 + 1 group)
                    name = self.residue[j]
                    self.__rpaa[i].append(RankProbaa.RankProbaa())
                    self.__rpaa[i][j].set_probaa(0.0)
                    self.__rpaa[i][j].set_rankaa(name)
                for j in range(7):
                    self.__prob6aa[i].append(0.0)  # Initialize the probability of Amino Acids (Initial Value 0.0, 6+1)
        except:
            self.__available = False
            print("出现错误")

        # The number of occurrences of each amino acid in the sequence is included in the dictionary (20 + 1 group)
        # and the list (6 + 1 group).
        try:
            sindex = 0
            for i in range(self.__sequence_number):
                self.__seq_tmp.append([])
            self.__sequence_number = 0
            self.__available = False
            if self.__fileFASTA is not None:
                f = open(self.__fileFASTA)
                for line in f:
                    for i in line:
                        if i is '>':
                            self.__available = True
                            self.__sequence_number += 1
                            self.__name_array.append(line.replace('\n', '').strip())
                            sindex = 0
                            break
                        if i is not '\n':
                            if i is 'S':
                                self.__rpaa[sindex][15].add()
                                self.__prob6aa[sindex][4] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'L':
                                self.__rpaa[sindex][9].add()
                                self.__prob6aa[sindex][0] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'A':
                                self.__rpaa[sindex][0].add()
                                self.__prob6aa[sindex][1] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'G':
                                self.__rpaa[sindex][5].add()
                                self.__prob6aa[sindex][3] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'K':
                                self.__rpaa[sindex][8].add()
                                self.__prob6aa[sindex][5] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'V':
                                self.__rpaa[sindex][17].add()
                                self.__prob6aa[sindex][0] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'T':
                                self.__rpaa[sindex][16].add()
                                self.__prob6aa[sindex][1] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'D':
                                self.__rpaa[sindex][2].add()
                                self.__prob6aa[sindex][2] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'E':
                                self.__rpaa[sindex][3].add()
                                self.__prob6aa[sindex][2] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'P':
                                self.__rpaa[sindex][12].add()
                                self.__prob6aa[sindex][3] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'N':
                                self.__rpaa[sindex][11].add()
                                self.__prob6aa[sindex][4] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'R':
                                self.__rpaa[sindex][14].add()
                                self.__prob6aa[sindex][5] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'F':
                                self.__rpaa[sindex][4].add()
                                self.__prob6aa[sindex][0] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'I':
                                self.__rpaa[sindex][7].add()
                                self.__prob6aa[sindex][0] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'Q':
                                self.__rpaa[sindex][13].add()
                                self.__prob6aa[sindex][4] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'Y':
                                self.__rpaa[sindex][19].add()
                                self.__prob6aa[sindex][0] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'C':
                                self.__rpaa[sindex][1].add()
                                self.__prob6aa[sindex][0] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'H':
                                self.__rpaa[sindex][6].add()
                                self.__prob6aa[sindex][5] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'M':
                                self.__rpaa[sindex][10].add()
                                self.__prob6aa[sindex][0] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is 'W':
                                self.__rpaa[sindex][18].add()
                                self.__prob6aa[sindex][0] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is '-':
                                self.__rpaa[sindex][20].add()
                                self.__prob6aa[sindex][6] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append(i)
                            elif i is '.':
                                self.__rpaa[sindex][20].add()
                                self.__prob6aa[sindex][6] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append('-')
                            elif i is 'X':
                                self.__rpaa[sindex][20].add()
                                self.__prob6aa[sindex][6] += 1.0
                                self.__seq_tmp[self.__sequence_number - 1].append('-')
                            else:
                                self.__rpaa[sindex][20].add()
                                self.__prob6aa[sindex][6] += 1.0  # 已验证
                                self.__seq_tmp[self.__sequence_number - 1].append('-')
                            sindex += 1
                f.close()
        except:
            self.__available = False
            print("出现错误")

        self.calculate_entropy()

        for i in range(self.__residus_number):
            self.__freqaa.append([])
            for j in range(21):
                self.__freqaa[i].append(self.__rpaa[i][j].get_probaa())

        for i in range(self.__residus_number):
            self.__freq6aa.append([])
            for j in range(7):
                self.__freq6aa[i].append(self.__prob6aa[i][j] / self.__sequence_number)

        for i in range(self.__residus_number):
            self.__gap_20.append([])
            for j in range(21):
                self.__gap_20[i].append(self.__rpaa[i][j].get_probaa() / self.__sequence_number)

        for i in range(self.__sequence_number):
            self.__seq.append(self.__seq_tmp[i])

    # Calculate the length of the amino acid sequence in the incoming file
    # and modify the number of amino acid sequences at the same time
    def calculate_res_number(self):
        if self.__fileFASTA is not None:
            self.__sequence_number = 0
            seq = {}  # Obtain amino acid sequence dictionary {"Name": "Sequence"}
            f = open(self.__fileFASTA)
            for line in f:
                if line.startswith('>'):
                    name = line.replace('>', '').split()[0]  # Obtain the name of amino acid sequence
                    seq[name] = ''
                    self.__sequence_number += 1
                else:
                    seq[name] += line.replace('\n', '').strip()  # Eliminate line breaks in a sequence
            f.close()
            b = list(seq.values())
            return len(b[0])

    def calculate_entropy(self):
        for i in range(self.__residus_number):
            self.__entropy_0.append(2.995373)
            self.__entropy_1.append(2.995373)
            for j in range(21):
                if self.__rpaa[i][j].get_probaa() > 0.0 and j != 20:
                    self.__entropy_0[i] += self.entropy_calc(self.__rpaa[i][j].get_probaa(),
                                                             self.__sequence_number,
                                                             self.__rpaa[i][20].get_probaa())
            for j in range(6):
                if self.__prob6aa[i][j] > 0.0:
                    self.__entropy_1[i] += self.simpilified_entropy_calc(self.__prob6aa[i][j],
                                                                         self.__sequence_number,
                                                                         self.__prob6aa[i][6], 6)

    def entropy_calc(self, probaa, seqnumber, probaa20):
        tmp = probaa / (seqnumber - probaa20)
        retval = tmp * math.log(tmp)
        return retval

    def simpilified_entropy_calc(self, prob6aa, seqnumber, pro6aa6, numberofgroup):
        tmp = prob6aa / (seqnumber - pro6aa6)
        retval = math.log(20.0) / math.log(numberofgroup) * tmp * math.log(tmp)
        return retval

    def re_calculate(self):
        for i in range(self.__residus_number):
            for j in range(self.__number_of_group + 1):
                self.__prob6aa[i][j] = 0.0
        for k in range(self.__residus_number):
            for i in range(21):
                for j in range(self.__number_of_group + 1):
                    if self.__group[i] == j:
                        self.__prob6aa[k][j] += self.__rpaa[k][i].get_probaa()
        for i in range(self.__residus_number):
            self.__entropy_1[i] = 2.99573
            for j in range(self.__number_of_group):
                if self.__prob6aa[i][j] > 0.0:
                    self.__entropy_1[i] += self.simpilified_entropy_calc(self.__prob6aa[i][j],
                                                                         self.__sequence_number,
                                                                         self.__prob6aa[i][self.__number_of_group],
                                                                         self.__number_of_group)
        self.__freq6aa = []
        for i in range(self.__residus_number):
            self.__freq6aa.append([])
            for j in range(7):
                self.__freq6aa[i].append(self.__prob6aa[i][j] / self.__sequence_number)

    def get_seq_number(self):
        return self.__sequence_number

    def get_res_number(self):
        return self.__residus_number

    def get_probaa(self):
        return self.__rpaa

    def get_freqaa(self):
        return self.__freqaa

    def get_freq6aa(self):
        return self.__freq6aa

    def get_prob6aa(self):
        return self.__prob6aa

    def get_entropy_0(self):
        return self.__entropy_0

    def get_entropy_1(self):
        return self.__entropy_1

    def set_reference(self, reference):
        self.__reference = reference

    def get_reference(self):
        return self.__reference

    def get_seq(self):
        return self.__seq

    def get_seq_names(self):
        return self.__name_array

    def display_seq(self):
        for i in range(self.__sequence_number):
            print("\nsequence{}:".format(i))
            for j in range(self.__residus_number):
                print(self.__seq[i][j], end="")

    def get_repseq(self):
        repseq = []
        for i in range(self.__residus_number):
            repseq.append(self.__seq[0][i])
        return repseq

    def get_reference_sequence(self):
        reference_string = ''
        sequence = self.get_seq()
        sequence_line = sequence[self.__reference]
        for c in sequence_line:
            if c is not '-':
                reference_string += c
        return reference_string

    # -------------------------------------
    def calculate_site(self, site):
        site_string = ''
        if self.get_seq_names() is not None:
            sequence_name = self.get_seq_names()[self.__reference]
            start = sequence_name.index("/")
            end = sequence_name.index("-")
            start_site = 1
            if end > start:
                start_site = int(sequence_name[start + 1, end])
            sequence = self.get_seq()
            sequence_line = sequence[self.__reference]
            actual_site = start_site
            for i in range(site):
                if sequence_line[i] is not '-':
                    actual_site = actual_site + 1
            site_string = str(actual_site) + str(sequence_line[site])
        return site_string

    def get_reference_start(self):
        start_site = 0
        if self.get_seq_names() is not None:
            sequence_name = self.get_seq_names()[self.__reference]
            start = sequence_name.index("/")
            end = sequence_name.index("-")
            start_site = int(sequence_name[start + 1, end])
        return start_site

    def get_actual_site(self, site):
        site_string = ''
        if self.get_seq_names() is not None:
            sequence_name = self.get_seq_names()[self.__reference]
            start = sequence_name.index("/")
            end = sequence_name.index("-")
            start_site = int(sequence_name[start + 1, end])
            sequence = self.get_seq()
            sequence_line = sequence[self.__reference]
            actual_site = start_site
            for i in range(site):
                if sequence_line[i] is not '-':
                    actual_site = actual_site + 1
            site_string = str(actual_site)
        return site_string

    # -------------------------------------

    def get_gap_20(self):
        return self.__gap_20

    def get_gap_6(self):
        return self.__freq6aa

    def set_gap_percent(self, p):
        self.__gap_percent = p

    def get_gap_percent(self):
        return self.__gap_percent

    def set_p_value_1(self, p_1):
        self.__pvalue_1 = p_1

    def get_p_value_1(self):
        return self.__pvalue_1

    def set_p_value_2(self, p_2):
        self.__pvalue_2 = p_2

    def get_p_value_2(self):
        return self.__pvalue_2

    def get_avaliable(self):
        return self.__available

    def get_group(self):
        return self.__group

    def set_group(self, group):
        self.__group = group

    def get_default(self):
        return self.__default_group

    def set_default(self, default_group):
        self.__default_group = default_group

    def set_default_group(self):
        self.__number_of_group = 6
        self.__group_names = ["hydrophobic", "AT", "negative", "conformational", "polar", "positive", "gap"]
        self.__group = [1, 0, 2, 2, 0, 3, 5, 0, 5, 0, 0, 4, 3, 4, 5, 4, 1, 0, 0, 0, 6]
        self.re_calculate()

    def set_group_4(self):
        self.__number_of_group = 4
        self.__group_names = ["non polar", "uncharged polar", "acidic", "basic", "gap"]
        self.__group = [0, 1, 2, 2, 0, 0, 3, 0, 3, 0, 0, 1, 0, 1, 3, 1, 1, 0, 0, 1, 4]
        self.re_calculate()

    def set_group_3(self):
        self.__number_of_group = 3
        self.__group_names = ["essential", "non-essential", "conditionally essential", "gap"]
        self.__group = [1, 2, 1, 1, 0, 2, 0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 2, 3]
        self.re_calculate()

    def set_group_7(self):
        self.__number_of_group = 7
        self.__group_names = ["aliphatic", "aromatic", "acidic", "basic", "hydroxylic", "sulfur-containing", "amidic",
                              "gap"]
        self.__group = [0, 5, 2, 2, 1, 0, 3, 0, 3, 0, 5, 6, 0, 6, 3, 4, 4, 0, 1, 1, 7]
        self.re_calculate()

    def get_number_of_group(self):
        return self.__number_of_group

    def set_number_of_group(self, number):
        self.__number_of_group = number

    def get_group_names(self):
        return self.__group_names

    def set_group_names(self, group_names):
        self.__group_names = group_names

    def char_to_byte(self, c):
        b = -1
        for i in range(len(self.residue)):
            if c == self.residue[i] or c == self.residue_small[i]:
                b = i
        if c == '.':
            b = 20
        return b

    def byte_to_char(self, b):
        if 0 <= b < len(self.residue):
            return self.residue_small[b]
        return '-'

    def print_sorted_inf_20(self):
        gap = self.get_gap_20()
        information_20 = self.get_entropy_0()
        seq = self.get_seq()
        temp = list(enumerate(information_20))
        temp = sorted(temp, key=lambda x: x[1], reverse=True)  # 进行排序并且返回原始索引（由大到小）
        sorted_origin_index = []
        for i in temp:
            sorted_origin_index.append(i[0])
        count = 1
        print("information (group 20+1):")
        for i in sorted_origin_index:
            if gap[i][20] > self.__gap_percent:
                continue
            else:
                print("{}{}: \t{:.3f}".format(i + 1, seq[0][i], math.ceil(information_20[i] * 1000) / 1000))
                count = count + 1

    # def print_sorted_inf_6(self):
    #     gap = self.get_gap_6()
    #     information_6 = self.get_entropy_1()
    #     seq = self.get_seq()
    #     temp = list(enumerate(information_6))
    #     temp = sorted(temp, key=lambda x: x[1], reverse=True)  # 进行排序并且返回原始索引（由大到小）
    #     sorted_origin_index = []
    #     for i in temp:
    #         sorted_origin_index.append(i[0])
    #     count = 1
    #     print("information (group {}+1):".format(self.__number_of_group))
    #     for i in sorted_origin_index:
    #         if gap[i][self.__number_of_group] > self.__gap_percent:
    #             continue
    #         else:
    #             print("{}{}: \t{:.3f}".format(i + 1, seq[0][i], math.ceil(information_6[i] * 1000) / 1000))
    #             count = count + 1

    def print_gap_percent_20(self):
        gap = self.get_gap_20()
        print("gap percent:")
        count = 1
        for i in gap:
            print("{}:  \t{:.3f}".format(count, i[20]))
            count = count + 1

    # def print_gap_percent_6(self):
    #     gap = self.get_gap_6()
    #     print("gap percent:")
    #     count = 1
    #     for i in gap:
    #         print("{}:  \t{:.3f}".format(count, i[6]))
    #         count = count + 1

    def inf_to_file_20(self, file):
        fp = open(file, 'w')
        gap = self.get_gap_20()
        information_20 = self.get_entropy_0()
        seq = self.get_seq()
        temp = list(enumerate(information_20))
        temp = sorted(temp, key=lambda x: x[1], reverse=True)  # 进行排序并且返回原始索引（由大到小）
        sorted_origin_index = []
        for i in temp:
            sorted_origin_index.append(i[0])
        count = 1
        fp.write("rank".ljust(6) + "position".ljust(10) + "information".ljust(8) + "\n")
        for i in sorted_origin_index:
            if gap[i][20] > self.__gap_percent:
                continue
            else:
                fp.write(str(count).ljust(6) + (str(i + 1) + str(seq[0][i])).ljust(10) +
                         str(math.ceil(information_20[i] * 1000) / 1000).ljust(8) + '\n')
                count = count + 1
        fp.write("count: {}".format(count))
        fp.close()

    # def inf_to_file_6(self, file):
    #     fp = open(file, 'w')
    #     gap = self.get_gap_6()
    #     information_6 = self.get_entropy_1()
    #     seq = self.get_seq()
    #     temp = list(enumerate(information_6))
    #     temp = sorted(temp, key=lambda x: x[1], reverse=True)  # 进行排序并且返回原始索引（由大到小）
    #     sorted_origin_index = []
    #     for i in temp:
    #         sorted_origin_index.append(i[0])
    #     count = 1
    #     for i in sorted_origin_index:
    #         if gap[i][6] > self.__gap_percent:
    #             continue
    #         else:
    #             fp.write("{}{}: \t{:.3f}\n".format(i + 1, seq[0][i], math.ceil(information_6[i] * 1000) / 1000))
    #             count = count + 1
    #     fp.write("count: {}".format(count))
    #     fp.close()

    def main(self):
        pass
