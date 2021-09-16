# -*- coding: utf-8 -*-
# @Time : 1/14/2020 7:47 PM
# @Author : Edgar Qian
# @Email : qianneng_se@163.com
# @File : Mulnfo6_out.py

from myProCon.ProCon import ProbabilityCalculator, MutualInformation

a = ProbabilityCalculator.ProbabilityCalculator("4503949.aligned")
b = MutualInformation.MutualInformation(a, 0.1, 0.01, 0.05, True)

mul_info = b.get_mut_info()
mul_info = list(map(str, mul_info))
residus_number = b.get_residus_number()

with open("mul_result_6", "w") as f:
    for i in range(residus_number):
        f.writelines(mul_info[i])
        f.write("\n")
