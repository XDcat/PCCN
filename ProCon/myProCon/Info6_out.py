# -*- coding: utf-8 -*-
# @Time : 1/14/2020 8:11 PM
# @Author : Edgar Qian
# @Email : qianneng_se@163.com
# @File : Info6_out.py

from myProCon import ProbabilityCalculator, MutualInformation, TripletFinder
import math

a = ProbabilityCalculator.ProbabilityCalculator("4503949.aligned")
b = MutualInformation.MutualInformation(a, 0.1, 0.01, 0.05, True)
gap = a.get_freqaa()
information_6 = a.get_entropy_1()
seq = a.get_seq()

temp = list(enumerate(information_6))
temp = sorted(temp, key=lambda x: x[1], reverse=True)   # 进行排序并且返回原始索引（由大到小）
sorted_origin_index = []
for i in temp:
    sorted_origin_index.append(i[0])

print("-----------------gap percent-----------------")
count = 0
for i in gap:
    print("{}:  \t{:.3f}".format(count, i[20]))
    count = count + 1

print("-----------------information-----------------")
count = 0
for i in sorted_origin_index:
    if gap[i][20] > 0.1:
        continue
    else:
        print("{}{}: \t{:.3f}".format(i+1, seq[0][i], math.ceil(information_6[i] * 1000) / 1000))
        count = count + 1
print("count:{}".format(count))