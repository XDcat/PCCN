# -*- coding: utf-8 -*-
# @Time : 2020/4/26 17:28
# @Author : Edgar Qian
# @Email : qianneng_se@163.com
# @File : test.py

from myProCon import ProbabilityCalculator, MutualInformation, TripletFinder

# a = ProbabilityCalculator.ProbabilityCalculator("D:/Downloads/ProCon-v2.0/examples/example-input.txt")
a = ProbabilityCalculator.ProbabilityCalculator(r"D:\Box\python\Cov2_protein\ProCon\myProCon\4503949.aligned")
b = MutualInformation.MutualInformation(a)
c = TripletFinder.TripletFinder(b)
c.show_graph("Triplets found among the covariant pairs")
