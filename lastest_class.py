#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 09:51:04 2018

@author: huiyuzhang
"""
import pandas as pd
import os
import re
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class classifier_workshop():
    def walk_data(self,tmp_path):
        temp_data=pd.DataFrame()
        #the_path="/Users/huiyuzhang/Desktop/workshop_data"
        for root, dirs, files in os.walk(tmp_path):
            for filename in files:
                #print(filename)
                #print(root+'/'+filename)
                tmp_label=root.split('/')
                tmp_label=tmp_label[-1]
                tmp_path=root+'/'+filename
                temp_data=temp_data.append({'path':tmp_path,
                                          'label':tmp_label},ignore_index=True)
        return(temp_data)
    def clean_data(self,df):
        stop_words=set(stopwords.words('english'))
        df['body']=None   
        for index, the_file in zip(df.index,df.path):
            tmp_read=open(the_file,'r',errors='replace')
            tmp_read_var=tmp_read.readlines()
            tmp_read_var=' '.join(tmp_read_var)
            tmp_read_var=re.sub('[^A-Za-z]+',' ',tmp_read_var)
            tmp_read_var=re.sub(' +',' ',tmp_read_var)
            tmp_read_var=tmp_read_var.split()
            tmp_read_var=[word.lower() for word in tmp_read_var]
            tmp_read_var=[word for word in tmp_read_var if word not in stop_words]
            tmp_read_var=' '.join(tmp_read_var)
            if len(tmp_read_var) !=0:
                df.loc[index].body=tmp_read_var
            #df.loc[index].body=tmp_read_var
            tmp_read.close()
        df=df.drop('path',1)
        df=df.dropna()
        return(df)
        
    def train(self,df,model):
        vectorizer=TfidfVectorizer(max_features=1000,ngram_range=(1,3))
            
        tdm=pd.DataFrame(vectorizer.fit_transform(df.body).toarray())
        
        tdm.columns=vectorizer.get_feature_names()
        
        label_enc=preprocessing.LabelEncoder()
        
        #label_enc.fit_transform(pd_data.label)
        the_labels=label_enc.fit_transform(df.label)
        
        model.fit(tdm,the_labels)
        
        return(model,vectorizer,label_enc)
        
    def classifier(self,m,v,l,thresh_tmp,tmp_text):

        the_sample=v.transform([tmp_text])
        #the_val=model.predict(the_sample)
        #label_enc.inverse_transform(the_val)

        #the_val=model.predict([tdm.loc[258]])
        #label_enc.inverse_transform(the_val)
        #tdm.loc[1]
        the_prob=m.predict_proba(the_sample)
        
        the_raw=m.classes_
        
        the_pred=pd.DataFrame({'prob':the_prob[0],'labels':the_raw})
        
        the_pred=pd.DataFrame({'prob':the_prob[0],'labels':l.inverse_transform(the_raw)})

        the_max_val=np.max(the_pred.prob)
        
        if(the_max_val >=thresh_tmp):
            the_confident_tmp=the_pred.prob==the_max_val
            the_predicted_value=the_pred[the_confident_tmp]
            the_pred_val=the_predicted_value
        else:
            the_pred_val='****Alert DO NOT MEET THRESHOLD****'+str(thresh_tmp)+' '+'UNABLE TO CLASSIFY'       
            

        #the_confident_tmp=the_pred.prob>=thresh
        #the_predicted_value=the_pred[the_confident_tmp]
        
        return(the_pred_val,the_pred)
     
               