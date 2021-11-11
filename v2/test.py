# -*- coding:utf-8 -*-
'''
__author__ = 'XD'
__mtime__ = 2021/11/11
__project__ = Cov2_protein
Fix the Problem, Not the Blame.
'''

import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    tips = sns.load_dataset("tips")
    print(tips)
    ax = sns.boxplot(x=tips["total_bill"])

    plt.show()