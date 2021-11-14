# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 12:16:32 2021

@author: anil
"""

	

	
	
#app.py

import pandas as pd 
import numpy as np 
import pickle
from flask import Flask, flash, request, redirect, url_for, render_template


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

data=pd.read_csv(r'D:\data science projects\lead status project\lemmatize_lead_status_data.csv')

# information
x=data.status_information
y=data.status
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=7)
tvec=TfidfVectorizer(ngram_range=(1,3),stop_words='english').fit(x_train)

#location
x_location=data.location
y_location=data.status
x_location_train,x_location_test,y_location_train,y_location_test=train_test_split(x_location,y_location,test_size=0.33,random_state=7)
tvec_location=TfidfVectorizer(ngram_range=(1,3),stop_words='english').fit(x_location_train)


#executive name
x_executive_name=data.executive_name
y_executive_name=data.status
x_executive_name_train,x_executive_name_test,y_executive_name_train,y_executive_name_test=train_test_split(x_executive_name,y_executive_name,test_size=0.33,random_state=7)
tvec_executive_name=TfidfVectorizer(ngram_range=(1,3),stop_words='english').fit(x_executive_name_train)



with open('lead_status_pkl' , 'rb') as f :
    model1 = pickle.load(f)
info1=tvec.transform(['not intrested'])
print(model1.predict(info1))



with open('lead_status_pkl2' , 'rb') as f :
    model2 = pickle.load(f)
info1=tvec.transform(['ringing no response intrested in data science'])
location1=tvec_location.transform(['hyderabad'])
executive_name1=tvec_executive_name.transform(['mohan'])
from scipy.sparse import hstack
test=hstack((info1,location1,executive_name1))
print(model2.predict(test))




 
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def msg_prediction():
    message = request.form['message']
    print(message)
    info1=tvec.transform([message])
    print(model1.predict(info1))
    my_prediction=model1.predict(info1)[0].upper()
    
    return render_template('result.html',prediction = my_prediction)


@app.route('/predict',methods=['POST'])
def msg_name_loc_prediction():
    message = request.form['message']
    location = request.form['location']
    executive_name = request.form['executive_name']

    print(message)
    info2=tvec.transform([message])
    location1=tvec_location.transform([location])
    executive_name=tvec_executive_name.transform([executive_name])    
    
    test=hstack((info2,location1,executive_name1))
    
    my_prediction2=model2.predict(test)[0].upper()
    
    return render_template('result.html',prediction = my_prediction2)




 
if __name__ == "__main__":
    app.run()