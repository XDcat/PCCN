# -*- coding: utf-8 -*-
# @Time : 2020/4/26 17:28
# @Author : Edgar Qian
# @Email : qianneng_se@163.com
# @File : test.py

from myProCon import ProbabilityCalculator, MutualInformation, TripletFinder

# a = ProbabilityCalculator.ProbabilityCalculator("D:/Downloads/ProCon-v2.0/examples/example-input.txt")
# a = ProbabilityCalculator.ProbabilityCalculator(r"D:\Box\python\Cov2_protein\ProCon\myProCon\4503949.aligned")
# a = ProbabilityCalculator.ProbabilityCalculator(r"D:\Box\python\Cov2_protein\ProCon\myProCon\4503949.aligned")
# a = ProbabilityCalculator.ProbabilityCalculator(r"D:\Box\python\Cov2_protein\data\test.aligned")
a = ProbabilityCalculator.ProbabilityCalculator(r"D:\Box\python\Cov2_protein\data\protein-matching-IPR042578.filter.fasta.aligned")
print(a.get_seq_number())
print(a.get_seq_names())
print(sorted(a.get_entropy_0()))
# print(a.get_entropy_0(), len(a.get_entropy_0()))
b = MutualInformation.MutualInformation(a)
print("get_mut_inf")
print(b.get_mut_inf(), len(b.get_mut_inf()))
c = TripletFinder.TripletFinder(b)
c.show_graph("Triplets found among the covariant pairs")
