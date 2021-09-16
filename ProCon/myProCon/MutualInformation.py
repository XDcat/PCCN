# -*- coding: utf-8 -*-
# @Time : 2019/8/9 17:28
# @Author : Edgar Qian
# @Email : qianneng_se@163.com
# @File : MutualInformation.py

import math
from myProCon import ProbabilityCalculator, MutObjNew


class MutualInformation:
    __group_flag = False
    __g_percent = 0.0
    __group = []
    __p_1 = 0.0
    __p_2 = 0.0
    __seq = []
    __mutuinf = []
    __residus_number = 0
    __sequence_number = 0
    __group_number = 0
    __number_of_pairs = 0
    __prob_prob = []
    __paraml = 0.0
    __max_muinf = 0.0
    __min_muinf = 0.0
    __ave_muinf = 0.0
    __stand_dev = 0.0
    __num_of_less_gap = 0
    __num_of_muincal = 0
    __kk_estimate = 0
    __top_num = 0
    __freqaa = []
    __list = []
    __list_p1 = []
    __list_p2 = []
    __group_table = []
    __l_table = []
    __mut_obj = []

    def __init__(self, pro_calc):
        self.__l_table = [0, -1, 1, 2, 3, 4, 5, 6, 7, -1, 8, 9, 10, 11, -1, 12,
                          13, 14, 15, 16, -1, 17, 18, -1, 19, -1, -1, -1, -1, -1]
        self.__p_1 = pro_calc.get_p_value_1()
        self.__p_2 = pro_calc.get_p_value_2()
        self.pro_calc = pro_calc
        self.__seq = pro_calc.get_seq()
        self.__g_percent = pro_calc.get_gap_percent()
        self.__residus_number = pro_calc.get_res_number()
        self.__sequence_number = pro_calc.get_seq_number()
        self.__group = pro_calc.get_group()
        for i in range(30):
            self.__group_table.append(-1)
        for i in range(len(pro_calc.residue) - 1):
            self.__group_table[ord(pro_calc.residue[i]) - ord('A')] = self.__group[i]
        self.__paraml = self.__sequence_number * self.__g_percent
        self.__number_of_pairs = self.__residus_number * self.__residus_number / 2
        for i in range(self.__residus_number):
            self.__mutuinf.append([])
            for j in range(self.__residus_number):
                self.__mutuinf[i].append(0.0)
        self.__freqaa = pro_calc.get_prob6aa()
        self.__group_number = pro_calc.get_number_of_group()
        print("number of group = {}".format(self.__group_number))
        self.__ave_muinf = 0.0
        self.__stand_dev = 0.0
        self.__num_of_less_gap = 0
        self.__kk_estimate = 0
        for i in range(self.__residus_number):
            if self.__freqaa[i][self.__group_number] < self.__paraml:
                self.__num_of_less_gap += 1
            for j in range(i + 1, self.__residus_number):
                if self.__freqaa[i][self.__group_number] < self.__paraml and \
                        self.__freqaa[j][self.__group_number] < self.__paraml:
                    self.__kk_estimate += 1
        print("kkEstimate = {}".format(self.__kk_estimate))
        # self.__mut_obj = [[Muinf12, Site1, Site2], [~, ~, ~], ... ]
        for i in range(self.__kk_estimate):
            self.__mut_obj.append(MutObjNew.MutObjNew())
            self.__mut_obj[i].set_mu_inf_12(0.0)
            self.__mut_obj[i].set_site_1(0)
            self.__mut_obj[i].set_site_2(0)
        self.__num_of_muincal = 0
        probmutu_tmp = []
        for i in range(self.__group_number):
            self.__prob_prob.append([])
            probmutu_tmp.append([])
            for k in range(self.__group_number):
                self.__prob_prob[i].append(0)
                probmutu_tmp[i].append(0)
        for i in range(self.__residus_number):
            for j in range(i + 1, self.__residus_number):
                numnogap = self.__sequence_number
                # When qualified, set the values in both binary lists to 0.0
                if self.__freqaa[i][self.__group_number] < self.__paraml and \
                        self.__freqaa[j][self.__group_number] < self.__paraml:
                    for ii in range(self.__group_number):
                        for jj in range(self.__group_number):
                            self.__prob_prob[ii][jj] = 0.0
                            probmutu_tmp[ii][jj] = 0.0
                    for k in range(self.__sequence_number):
                        if self.__seq[k][j] == '-' or self.__seq[k][i] == '-':
                            numnogap -= 1.0
                        else:
                            ii = self.__group_table[ord(self.__seq[k][i]) - ord('A')]
                            jj = self.__group_table[ord(self.__seq[k][j]) - ord('A')]
                            if ii >= 0 and jj >= 0:
                                self.__prob_prob[ii][jj] += 1.0
                            elif ii < 0:
                                print("symbol {} is not include in 20 AAs or gap(-)".format(self.__seq[k][i]))
                            elif jj < 0:
                                print("symbol {} is not include in 20 AAs or gap(-)".format(self.__seq[k][j]))
                    numnogap_tmp = 1.0 / numnogap
                    for ii in range(self.__group_number):
                        for jj in range(self.__group_number):
                            if self.__prob_prob[ii][jj] != 0.0:
                                a = self.__prob_prob[ii][jj] * numnogap_tmp
                                b = self.__prob_prob[ii][jj] * numnogap
                                probmutu_tmp[ii][jj] = self.prob_mutu_calc(a, b, self.__freqaa[i][ii],
                                                                           self.__freqaa[j][jj])
                                self.__mut_obj[self.__num_of_muincal].set_mu_inf(ii, jj, probmutu_tmp[ii][jj])  # !
                                if probmutu_tmp[ii][jj] >= 40:
                                    pass
                                    # print("numOfMuInCal = {}; i = {}; j = {}; probmutuTmp[{}][{}]={}"
                                    #       .format(self.__num_of_muincal, i, j, ii, jj, probmutu_tmp[ii][jj]))
                                self.__mutuinf[i][j] += probmutu_tmp[ii][jj]  # 已验证
                    if self.__mutuinf[i][j] > 0:
                        self.__mut_obj[self.__num_of_muincal].set_site_1(i)
                        self.__mut_obj[self.__num_of_muincal].set_site_2(j)
                        self.__mut_obj[self.__num_of_muincal].set_mu_inf_12(self.__mutuinf[i][j])
                        self.__ave_muinf += self.__mutuinf[i][j]
                        self.__num_of_muincal += 1
        self.__ave_muinf /= self.__num_of_muincal
        self.__mut_obj.sort(key=self.take_mu_inf_12, reverse=True)
        self.__max_muinf = self.__mut_obj[0].get_mu_inf_12()
        self.__min_muinf = self.__mut_obj[(len(self.__mut_obj)) - 1].get_mu_inf_12()
        print("max is between site {} and site {} max = {}"
              .format(self.__mut_obj[0].get_site_1() + 1, self.__mut_obj[0].get_site_2() + 1, self.__max_muinf))
        print("min is between site {} and site {} min = {}"
              .format(self.__mut_obj[(len(self.__mut_obj) - 1)].get_site_1() + 1,
                      self.__mut_obj[(len(self.__mut_obj) - 1)].get_site_2() + 1, self.__min_muinf))
        print("ave = {}".format(self.__ave_muinf))
        rest = self.__num_of_muincal % 4
        for i in range(rest):
            self.__stand_dev += (self.__mut_obj[i].get_mu_inf_12() - self.__ave_muinf) * \
                                (self.__mut_obj[i].get_mu_inf_12() - self.__ave_muinf) / (self.__num_of_muincal - 1)
        print("numOfMuInCal = {}; standdev = {}".format(self.__num_of_muincal, self.__stand_dev))
        for i in range(rest, self.__num_of_muincal, 4):
            self.__stand_dev += (self.__mut_obj[(i + 0)].get_mu_inf_12() - self.__ave_muinf) * \
                                (self.__mut_obj[(i + 0)].get_mu_inf_12() - self.__ave_muinf) / \
                                (self.__num_of_muincal - 1)
            self.__stand_dev += (self.__mut_obj[(i + 1)].get_mu_inf_12() - self.__ave_muinf) * \
                                (self.__mut_obj[(i + 1)].get_mu_inf_12() - self.__ave_muinf) / \
                                (self.__num_of_muincal - 1)
            self.__stand_dev += (self.__mut_obj[(i + 2)].get_mu_inf_12() - self.__ave_muinf) * \
                                (self.__mut_obj[(i + 2)].get_mu_inf_12() - self.__ave_muinf) / \
                                (self.__num_of_muincal - 1)
            self.__stand_dev += (self.__mut_obj[(i + 3)].get_mu_inf_12() - self.__ave_muinf) * \
                                (self.__mut_obj[(i + 3)].get_mu_inf_12() - self.__ave_muinf) / \
                                (self.__num_of_muincal - 1)
        self.__stand_dev = math.sqrt(self.__stand_dev)
        print("standdev = {}".format(self.__stand_dev))
        limit_p1 = self.__ave_muinf + self.__stand_dev * self.get_z_value_1(self.__p_2)
        limit_p2 = self.__ave_muinf + self.__stand_dev * self.get_z_value_2(self.__p_2)
        self.__list_p1 = []
        self.__list_p2 = []
        loop = 0  # !
        while self.__mut_obj[loop].get_mu_inf_12() > limit_p2:
            self.__list.append(self.__mut_obj[loop])
            if self.__mut_obj[loop].get_mu_inf_12() > limit_p1:
                self.__list_p1.append(self.__mut_obj[loop])
            else:
                self.__list_p2.append(self.__mut_obj[loop])
            loop += 1
        self.__top_num = loop

    def take_mu_inf_12(self, elem):
        """
        :param elem:self.__mut_obj
        :return:self.mut_obj.get_mu_inf_12()
        """
        return elem.get_mu_inf_12()

    def get_z_value_1(self, p_value_1):
        z_value_1 = 0.0
        if p_value_1 == 0.00001:
            z_value_1 = 4.25
        elif p_value_1 == 0.00005:
            z_value_1 = 3.88
        elif p_value_1 == 0.0001:
            z_value_1 = 3.7
        elif p_value_1 == 0.0005:
            z_value_1 = 3.3
        elif p_value_1 == 0.001:
            z_value_1 = 3.1
        elif p_value_1 == 0.002:
            z_value_1 = 2.88
        elif p_value_1 == 0.003:
            z_value_1 = 2.75
        elif p_value_1 == 0.004:
            z_value_1 = 2.65
        elif p_value_1 == 0.005:
            z_value_1 = 2.58
        elif p_value_1 == 0.01:
            z_value_1 = 2.33
        elif p_value_1 == 0.012:
            z_value_1 = 2.26
        elif p_value_1 == 0.014:
            z_value_1 = 2.2
        elif p_value_1 == 0.015:
            z_value_1 = 2.17
        return z_value_1

    def get_z_value_2(self, p_value_2):
        z_value_2 = 0.0
        if p_value_2 == 0.016:
            z_value_2 = 2.14
        elif p_value_2 == 0.017:
            z_value_2 = 2.12
        elif p_value_2 == 0.018:
            z_value_2 = 2.1
        elif p_value_2 == 0.019:
            z_value_2 = 2.08
        elif p_value_2 == 0.02:
            z_value_2 = 2.06
        elif p_value_2 == 0.03:
            z_value_2 = 1.88
        elif p_value_2 == 0.04:
            z_value_2 = 1.75
        elif p_value_2 == 0.05:
            z_value_2 = 1.65
        return z_value_2

    def get_list_p1(self):
        return self.__list_p1

    def get_list_p2(self):
        return self.__list_p2

    def get_list(self):
        return self.__list

    def prob_mutu_calc(self, a, b, freqaaiii, freqaajjj):
        return a * a * 1000.0 * math.log(b / (freqaaiii * freqaajjj))

    def display_mutu_info(self):
        seq = self.pro_calc.get_seq()
        mul_list = []
        for i in range(len(self.__mutuinf)):
            pass
        for i in range(self.__residus_number):
            for j in range(self.__residus_number):
                mul_list.append([str(i + 1) + seq[0][i], str(j + 1) + seq[0][j], self.__mutuinf[i][j]])
        temp = sorted(mul_list, key=lambda x: x[2], reverse=True)  # 进行排序并且返回原始索引（由大到小）
        count = 1
        print("Rank".ljust(8) + "Site1".ljust(8) + "Site2".ljust(8) + "Mutual Information".ljust(8))
        for i in temp:
            if i[2] > 10:
                print("{}{}{}{:.3f}".format(str(count).ljust(8), i[0].ljust(8), i[1].ljust(8), i[2]))
                count += 1
            else:
                break

    def get_mut_inf(self):
        return self.__mutuinf

    def get_g_percent(self):
        return self.__g_percent

    def get_residus_number(self):
        return self.__residus_number

    def get_num_of_muincal(self):
        return self.__num_of_muincal

    def get_num_of_less_gap(self):
        return self.__num_of_less_gap

    def get_top_num(self):
        return self.__top_num

    def get_mut_obj(self):
        return self.__mut_obj

    def get_group_flag(self):
        return self.__group_flag

    def get_seq(self):
        return self.pro_calc.get_seq()

    def mut_inf_to_file(self, file):
        fp = open(file, 'w')
        seq = self.get_seq()
        mul_list = []
        for i in range(len(self.__mutuinf)):
            pass
        for i in range(self.__residus_number):
            for j in range(self.__residus_number):
                mul_list.append([str(i + 1) + seq[0][i], str(j + 1) + seq[0][j], self.__mutuinf[i][j]])
        temp = sorted(mul_list, key=lambda x: x[2], reverse=True)  # 进行排序并且返回原始索引（由大到小）
        count = 1
        fp.write("Rank".ljust(8) + "Site1".ljust(8) + "Site2".ljust(8) + "Mutual Information".ljust(8) + "\n")
        for i in temp:
            if i[2] > 10:
                fp.write("{}{}{}{:.3f}\n".format(str(count).ljust(8), i[0].ljust(8), i[1].ljust(8), i[2]))
                count += 1
            else:
                break
        fp.close()

    # mark: is never used
    def rank_muinfo(self):
        self.__max_muinf = self.__mut_obj[0][0]
        self.__min_muinf = self.__mut_obj[len(self.__mut_obj) - 1][0]
        print("max = {} min = {}".format(self.__max_muinf, self.__min_muinf))
        print("ave = {}".format(self.__ave_muinf))
        print("standdev = ".format(self.__stand_dev))
