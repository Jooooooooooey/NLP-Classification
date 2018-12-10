# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import sys
import os

the_path="/Users/huiyuzhang/Desktop/workshop_data"

tmp_data=pd.DataFrame()
for root, dirs, files in os.walk(the_path):
    for filename in files:
        #print(filename)
        #print(root+'/'+filename)
        tmp_label=root.split('/')
        tmp_label=tmp_label[-1]
        tmp_path=root+'/'+filename
        tmp_data=tmp_data.append({'path':tmp_path,
                         'label':tmp_label},ignore_index=True)
print(tmp_data)

        
