# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 11:27:27 2021

@author: brian
"""

import pandas as pd

#分類答案csv
out_ans = pd.read_csv("out4_IE_46-2.csv", header = None)
#偵測字串座標csv
coor = pd.read_csv("submission_v4.csv", header = None)

print(coor)
out_ans.columns = [9]

#coor = coor.drop(9, axis = 1)

out_ans=pd.concat([coor, out_ans], axis = 1)
out_ans[1] = out_ans[1].astype(str).astype(int)
out_ans[2] = out_ans[2].astype(str).astype(int)
out_ans[3] = out_ans[3].astype(str).astype(int)
out_ans[4] = out_ans[4].astype(str).astype(int)
out_ans[5] = out_ans[5].astype(str).astype(int)
out_ans[6] = out_ans[6].astype(str).astype(int)
out_ans[7] = out_ans[7].astype(str).astype(int)
out_ans[8] = out_ans[8].astype(str).astype(int)
out_ans[9] = out_ans[9].astype(str)

print(out_ans)
print(out_ans.dtypes)
pd.DataFrame.to_csv(out_ans, "submission_out4_IE_46-2.csv", header=False, index=False)

