# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 22:40:47 2021

@author: brian
"""

import pandas as pd

out_ans = pd.read_csv("submission_v4_2.csv", header = None)
coor = pd.read_csv("submission_v4_2.csv", header = None)

out_ans = pd.DataFrame(out_ans)
#print(out_ans.dtypes)
#out_ans[7] = out_ans[7].astype(str).astype(int)
#out_ans[8] = out_ans[8].astype(str).astype(int)

#print(out_ans.dtypes)
#out_ans = out_ans.drop(9, axis = 1)

a=out_ans[1]
b=out_ans[2]
c=pd.concat([a, b], axis = 1)

a=out_ans[3]
b=out_ans[4]
d=pd.concat([a, b], axis = 1)

a=out_ans[5]
b=out_ans[6]
e=pd.concat([a, b], axis = 1)

a=out_ans[7]
b=out_ans[8]
f=pd.concat([a, b], axis = 1)

out_ans = out_ans.drop(1, axis = 1)
out_ans = out_ans.drop(2, axis = 1)
out_ans = out_ans.drop(3, axis = 1)
out_ans = out_ans.drop(4, axis = 1)
out_ans = out_ans.drop(5, axis = 1)
out_ans = out_ans.drop(6, axis = 1)
out_ans = out_ans.drop(7, axis = 1)
out_ans = out_ans.drop(8, axis = 1)

# g=out_ans[10]

# gg= pd.DataFrame(['經紀人'])
# g = g[:-1]
# g = pd.concat([gg , g], axis = 0)
# print(g)
# g = g.reset_index()
# g = g.drop('index' ,axis = 1)
# print(g)
#out_ans = out_ans.drop(10, axis = 1)

out_ans=pd.concat([out_ans, c, f, e, d], axis = 1)

print(out_ans)

pd.DataFrame.to_csv(out_ans, "submission_v4_2.csv", header=False, index=False)



#out_ans=pd.concat([coor, g], axis = 1)
#pd.DataFrame.to_csv(out_ans, "submission_rr.csv", header=False, index=False)

