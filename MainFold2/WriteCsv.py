# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 20:48:14 2021

@author: brian
"""

import json
import pandas as pd


with open("pri_json_3_adj.json", encoding="utf-8") as f:  
    data = json.load(f)
    
    all_ans = []
    for i in data: # 照片
        
        for j in data[i]: # 圖片內題目
            ans = []
            ans.append(i)
            for k in data[i][j]["new_str_coor"]:
            #for l in data[i][j]["new_str_coor"]:
                ans.append(int(float(k)))
            all_ans.append(ans)

#out_ans = pd.read_csv("output.csv")

 
new_df = pd.DataFrame(all_ans)
#new_df = pd.concat([new_df, out_ans], axis = 1)
#pd.DataFrame.to_csv(new_df, "submission.csv", header=False, index=False)

pd.DataFrame.to_csv(new_df, "submission_pri.csv", header=False, index=False)

