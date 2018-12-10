#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 09:55:53 2018

@author: huiyuzhang
"""
#import pandas as pd
#import os
#import re
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from lastest_class import classifier_workshop
#from sklearn import preprocessing
#from sklearn.feature_extraction.text import TfidfVectorizer


the_path='/Users/huiyuzhang/Desktop/workshop_data'
the_framework=classifier_workshop()
pd_data=the_framework.walk_data(the_path)
pd_data=the_framework.clean_data(pd_data)


################## Random Forest #########################
model_in=RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)
the_model_out, the_vec_out, the_label_enc_out=the_framework.train(pd_data,model_in)
thresh=0.30
my_text='fishing'
my_pred_value, checker=the_framework.classifier(the_model_out, the_vec_out,the_label_enc_out,thresh, my_text)

################### GradientBoosting #########################
model_in=GradientBoostingClassifier(n_estimators=100,max_depth=2,random_state=0)
the_model_out, the_vec_out, the_label_enc_out=the_framework.train(pd_data,model_in)
thresh=0.30
my_text='fishing'
my_pred_value, checker=the_framework.classifier(the_model_out, the_vec_out,the_label_enc_out,thresh, my_text)
#################### AdaBoostClassifier ########################
model_in=AdaBoostClassifier(n_estimators=100)
the_model_out, the_vec_out, the_label_enc_out=the_framework.train(pd_data,model_in)
thresh=0.30
my_text='fishing'
my_pred_value, checker=the_framework.classifier(the_model_out, the_vec_out,the_label_enc_out,thresh, my_text)





