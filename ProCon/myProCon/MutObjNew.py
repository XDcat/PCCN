# -*- coding: utf-8 -*-
# @Time : 2019/8/9 17:28
# @Author : Edgar Qian
# @Email : qianneng_se@163.com
# @File : MutObjNew.py

from . import MutInfo


class MutObjNew:
    def __init__(self):
        self.mu_inf = []
        self.mu_inf_12 = 0.0
        self.site_1 = 0
        self.site_2 = 0

    def set_mu_inf_12(self, mu_inf_12):
        self.mu_inf_12 = mu_inf_12

    def set_site_1(self, site_1):
        self.site_1 = site_1

    def set_site_2(self, site_2):
        self.site_2 = site_2

    def set_mu_inf(self, mu_inf):
        self.mu_inf = mu_inf

    def set_mu_inf(self, i, j, value):
        mut_info = MutInfo.MutInfo(i, j, value)
        self.mu_inf.append(mut_info)

    def get_mu_inf_12(self):
        return self.mu_inf_12

    def get_site_1(self):
        return self.site_1

    def get_site_2(self):
        return self.site_2

    def get_mu_inf(self):
        return self.mu_inf
