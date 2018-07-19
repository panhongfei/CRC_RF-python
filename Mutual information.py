#!/usr/bin/python
# -*- coding:utf-8 -*-

from minepy import MINE
import pandas as pd
import numpy as np

# path = r"D:\learn\Bioinfo\zzz_paper\huxinxi\char_select1.csv"
path = r"D:\learn\Bioinfo\zzz_paper\RF_program\RandomForest\RF119\new_old_H_C_223_519.csv"

data = pd.read_csv(path,header = 0, sep=",")
col_num = len(data.loc[0])
print col_num

m = MINE()
m_score = []

#n = m.compute_score(np.array(data[1]),np.array(data[2]))
# print n
for i in range(col_num - 1)[1:]:
    # for j in range(i,col_num):
    m.compute_score(np.array(data.iloc[:,i]), np.array(pd.Categorical(data.iloc[:,(col_num -1)]).codes))
    m_score.append(m.mic())
print m_score
#
# m_score1 = enumerate(m_score)
# m_score2 = {}
# for i,value in m_score1:
#     m_score2[i] = value
#
# print m_score2


