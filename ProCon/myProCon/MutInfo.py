# -*- coding: utf-8 -*-
# @Time : 2019/8/9 17:28
# @Author : Edgar Qian
# @Email : qianneng_se@163.com
# @File : MutInfo.py


class MutInfo:
    def __init__(self, residue_1, residue_2, info):
        self.residue_1 = residue_1
        self.residue_2 = residue_2
        self.info = info

    def set_mut_info(self, residue_1, residue_2, info):
        self.residue_1 = residue_1
        self.residue_2 = residue_2
        self.info = info

    def get_residue_1(self):
        return self.residue_1

    def get_residue_2(self):
        return self.residue_2

    def get_info(self):
        return self.info