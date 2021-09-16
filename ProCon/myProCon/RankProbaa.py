# -*- coding: utf-8 -*-
# @Time : 2019/8/9 17:28
# @Author : Edgar Qian
# @Email : qianneng_se@163.com
# @File : RankProbaa.py


class RankProbaa:
    def __init__(self):
        self.probaa = 0
        self.rankaa = ''

    def set_probaa(self, probaa):
        self.probaa = probaa

    def cal_probaa(self, probaa):
        self.probaa = probaa

    def get_probaa(self):
        return self.probaa

    def set_rankaa(self, rankaa):
        self.rankaa = rankaa

    def get_rankaa(self):
        return self.rankaa

    def add(self):
        self.probaa = self.probaa + 1