# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 17:27:19 2018

@author: George
"""

import pandas as pd
from sklearn import preprocessing



train = pd.read_csv(r"D:\Curso_data_science\Curso 3 - machine learning\semana 4\train.csv", encoding='latin1')
test = pd.read_csv(r"D:\Curso_data_science\Curso 3 - machine learning\semana 4\test.csv", encoding='latin1')

#setting names of features of interest
features_list = ['agency_name','violation_street_number','city','state', 'ticket_issued_date','disposition','hearing_date', 'fine_amount','late_fee', 'discount_amount', 'compliance']

#getting only features list plus complicance and setting Nan to 0
train = train[features_list]
train['compliance'].fillna(0, inplace=True)

#converting categorical to values
categorias = set(train['city'])|{'<unknown>'}
train['city']= pd.Categorical(train['city'],categories=categorias).fillna('<unknown>').codes
test['city']= pd.Categorical(test['city'],categories=categorias).fillna('<unknown>').codes

train['state'].fillna("<unknown>", inplace=True)
categorias2 = set(train['state'])|{'<unknown>'}
train['state']= pd.Categorical(train['state'],categories=categorias2).fillna('<unknown>').codes
test['state']= pd.Categorical(test['state'],categories=categorias2).fillna('<unknown>').codes

train['violation_street_number'].fillna("<unknown>", inplace=True)
categorias3 = set(train['violation_street_number'])|{'<unknown>'}
train['violation_street_number']= pd.Categorical(train['violation_street_number'],categories=categorias3).fillna('<unknown>').codes
test['violation_street_number']= pd.Categorical(test['violation_street_number'],categories=categorias3).fillna('<unknown>').codes


train['disposition'].fillna("<unknown>", inplace=True)
categorias4 = set(train['disposition'])|{'<unknown>'}
train['disposition']= pd.Categorical(train['disposition'],categories=categorias3).fillna('<unknown>').codes
test['disposition']= pd.Categorical(test['disposition'],categories=categorias3).fillna('<unknown>').codes

le = preprocessing.LabelEncoder()
train['agency_name'] = le.fit_transform(train['agency_name'])
test['agency_name'] = le.transform(test['agency_name'])

train.dropna(inplace=True)

#getting X and Y training form trin dataset
X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]


#getting X test with only selected features
X_test = test[[feature for feature in features_list if feature!='compliance']]



X_train['ticket_issued_date'] =  pd.to_datetime(X_train['ticket_issued_date'])
X_test['ticket_issued_date'] =  pd.to_datetime(X_test['ticket_issued_date'])

X_train['hearing_date'] =  pd.to_datetime(X_train['hearing_date'])
X_test['hearing_date'] =  pd.to_datetime(X_test['hearing_date'])


X_train['diff'] = X_train['hearing_date'] - X_train['ticket_issued_date']
X_test['diff'] = X_test['hearing_date'] - X_test['ticket_issued_date']

X_train['diff']= X_train['diff'].dt.days
X_test['diff']= X_test['diff'].dt.days


del X_train['ticket_issued_date']
del X_train['hearing_date' ]
del X_test['ticket_issued_date']
del X_test['hearing_date' ]



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train,y_train)


parametros = {'max_depth': [1,3,5,7,9,11,13,15,16,17,18,19,20] , 'max_features':[1,2,3,4,5]}
grid = GridSearchCV(RandomForestClassifier(),param_grid = parametros, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train2, y_train2)
grid.best_params_ 
grid.score(X_test2, y_test2)
grid.score(X_train, y_train)
X_test.fillna(0, inplace=True)
prob = grid.predict_proba(X_test)
proba = prob[:,1]
fim = pd.Series(data=proba, index = test['ticket_id'])



"""
parametros = {'penalty': ['l1', 'l2'] , 'C':[ 0.005, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(),param_grid = parametros, scoring='roc_auc')
grid.fit(X_train2, y_train2)
grid.best_params_ 

lr = LogisticRegression(penalty='l1',C=0.005).fit(X_train2, y_train2)
lr.score(X_train,y_train)
lr.score(X_test2,y_test2) 

lr = LogisticRegression(penalty='l1',C=0.005).fit(X_train, y_train)
X_test.fillna(0, inplace=True)
prob = lr.predict_proba(X_test)
proba = prob[:,1]
fim = pd.Series(data=proba, index = test['ticket_id'])

"""



