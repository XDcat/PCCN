# -*- coding: utf-8 -*-
# @Time : 2019/8/9 17:28
# @Author : Edgar Qian
# @Email : qianneng_se@163.com
# @File : MuiSite.py


class MuiSite:

    def __init__(self):
        self.site_j = []
        self.size_of_j = 0

    def set_size_of_j(self, size_of_j):
        self.size_of_j = size_of_j

    def get_size_of_j(self):
        return self.size_of_j

    def get_site_j(self):
        return self.site_j

    def init_site_j(self, length):
        for i in range(length):
            self.site_j.append(0)

    def set_site_j(self, position, value):
        self.site_j[position] = value

