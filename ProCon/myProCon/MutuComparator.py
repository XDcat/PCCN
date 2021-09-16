# -*- coding: utf-8 -*-
# @Time : 2019/8/9 17:28
# @Author : Edgar Qian
# @Email : qianneng_se@163.com
# @File : MutuComparator.py


class MutuComparator:
    def compare(self, p_first, p_second):
        a_first_weight = p_first.get_mu_inf_12()
        a_second_weight = p_second.get_mu_inf_12()
        diff = a_first_weight - a_second_weight
        if diff > 0.0:
            return -1
        if diff < 0.0:
            return 1
        return 0
