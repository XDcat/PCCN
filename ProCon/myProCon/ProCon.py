# -*- coding: utf-8 -*-
# @Time : 2019/10/10 9:43
# @Author : Edgar Qian
# @Email : qianneng_se@163.com
# @File : myProCon.py


from myProCon import ProbabilityCalculator
from myProCon import MutualInformation
from myProCon import TripletFinder


class ProCon:
    def __init__(self, file, gap_percent, p_1, p_2, group_flag=True):
        self.file = file
        self.gap_percent = gap_percent
        self.p_1 = p_1
        self.p_2 = p_2
        self.froup_flag = group_flag
        self.pc = ProbabilityCalculator.ProbabilityCalculator(file)
        self.mi = MutualInformation.MutualInformation(self.pc, gap_percent, p_1, p_2, group_flag)
        self.tf = TripletFinder.TripletFinder(self.mi)

    def get_seq_number(self):
        return self.pc.get_seq_number()

    def get_res_number(self):
        return self.pc.get_res_number()

    def get_probaa(self):
        return self.pc.get_probaa()

    def get_prob6aa(self):
        return self.pc.get_prob6aa()

    def get_freqaa(self):
        return self.pc.get_freqaa()

    def get_freq6aa(self):
        return self.pc.get_freq6aa()

    def get_mut_obj(self):
        return self.mi.get_mut_obj()

    def get_triplets(self):
        return self.tf.get_triplets()

