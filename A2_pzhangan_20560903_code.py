# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 21:21:01 2018

@author: Ashley
"""

import numpy as np
import pandas as pd
import seaborn as sns
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
import xgboost as xgb
'''
read data, preview data: including distribute, description stats
'''
print('=====load dataset=====')
train_set=pd.read_csv('D:\\18年\\bdt\\5002\\assign2\\trainFeatures.csv')
label=['income']
train_label=pd.read_csv('D:\\18年\\bdt\\5002\\assign2\\trainLabels.csv',names=label)
test_set=pd.read_csv('D:\\18年\\bdt\\5002\\assign2\\testFeatures.csv')
print('=====load dataset sucessfully=====')
print('=====go through the basic information of the dataset=====')
print('=====replace ? with defult value nan=====')
train_set.info()
train_set.head()
train_set.shape
des=train_set.describe()
des.fnlwgt
train_set.replace(' ?', np.nan, inplace=True)
test_set.replace(' ?', np.nan, inplace=True)
train_label.head()
train_set.isnull().sum()  #### blank in workclass,occupation and country
test_set.isnull().sum() ####similarly
train_label.sum()/train_label.count()

'''
when filling the defalt-values, focus on workclass, occupation and native-country:  nearly 10% data missed
Skewed data sets  also (consider to banlance the dataset)
'''
print(train_set.describe())
des
char_cols =['workclass', 'education', 'Marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
'''feature engineering'''
###firstlook at the corr
###没有多重共线性
corrmat = train_set.corr()
f,ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, vmax = .8, square = True)#heat plot
print('=====feature engineering=====')
#age
plt.hist(train_set['age'])
plt.show()
train_set['age'].describe()
#workclass
train_set['workclass'].fillna(' 0', inplace=True)
test_set['workclass'].fillna(' 0', inplace=True)
train_set['workclass'].value_counts()
train_set['workclass']=train_set['workclass'].str.replace('Never-worked', 'Without-pay')
train_set['workclass'].value_counts()
#fnlwgt
train_set['fnlwgt'].describe()####   large 
plt.hist(train_set['fnlwgt'])
plt.show()
train_set['fnlwgt'] = train_set['fnlwgt'].apply(lambda x: np.log1p(x))
test_set['fnlwgt'] = test_set['fnlwgt'].apply(lambda x: np.log1p(x))
#education
train_set['education'].describe()
test_set['education']=test_set['education'].str.replace('1st-4th', 'primary')
test_set['education']=test_set['education'].str.replace('Preschool', 'primary')
train_set['education']=train_set['education'].str.replace('Preschool', 'primary')
train_set['education']=train_set['education'].str.replace('1st-4th', 'primary')
train_set['education'].value_counts()
###education-num   not neccerary to change
train_set['education-num'].describe()
train_set['education-num'].value_counts()
plt.hist(train_set['education-num'])
plt.show()
##as shown in the picture,there exists a great gap in 12,so we add a feature : high-edu
def edu(edunum):
    if edunum>12:
        return 1
    else: 
        return 0 
train_set['highedu'] = train_set['education-num'].apply(edu)
test_set['highedu'] = test_set['education-num'].apply(edu)
#pd.pivot_table(train_set,index=['education','education-num'],values=['age'],aggfunc=[np.mean,len])
###check  if education and education contradict to each other-------no   coordinate
##marital-status
train_set['Marital-status']=train_set['Marital-status'].str.replace('Married-AF-spouse', 'Married-civ-spouse')
test_set['Marital-status']=test_set['Marital-status'].str.replace('Married-AF-spouse', 'Married-civ-spouse')
##occupation
train_set['occupation'].fillna(' 0', inplace=True)
test_set['occupation'].fillna(' 0', inplace=True)
#relationship\race      do not need change
###capital-gain,capital-loss     skewed data    may need to change latter
plt.hist(train_set['capital-gain'])
plt.show()
plt.hist(train_set['capital-loss'])
plt.show()

#work hour per week
plt.hist(train_set['hours-per-week'])
plt.show()
train_set['hours-per-week'].describe()

def workhour(hour):
    if hour>45:
        return 1
    else: 
        return 0 
train_set['workhard'] = train_set['hours-per-week'].apply(workhour)
test_set['workhard'] = test_set['hours-per-week'].apply(workhour)
train_set['workhard'].value_counts()

##native country  data seperate a lot,consider to add a feature merge them by continent or development level.
def native(country):
    if country in [' United-States',  ' 0']:
        return 'us'
    elif country in [' England', ' Germany', ' Canada', ' Italy', ' France', ' Greece', ' Philippines']:
        return 'western'
    elif country in [' Mexico', ' Cuba',' Puerto-Rico', ' Honduras', ' Jamaica', ' Columbia', ' Laos', ' Portugal', ' Haiti',
                     ' Dominican-Republic', ' El-Salvador', ' Guatemala', ' Peru', 
                     ' Trinadad&Tobago', ' Outlying-US(Guam-USVI-etc)', ' Nicaragua', ' Vietnam', ' Holand-Netherlands' ]:
        return 'developing' 
    elif country in [' India', ' Iran', ' Cambodia', ' Taiwan', ' Japan', ' Yugoslavia', ' China', ' Hong']:
        return 'eastern'
    elif country in [' South', ' Poland', ' Ireland', ' Hungary', ' Scotland', ' Thailand', ' Ecuador']:
        return 'polandteam'
    
    else: 
        return country    
train_set['development'] = train_set['native-country'].apply(native)
test_set['development'] = test_set['native-country'].apply(native)
print('=====dummy and standardize=====')

####view the data distributions again
sns.set()
cols = ['age', 'fnlwgt',  'education-num', 'development', 'highedu','capital-gain','capital-loss','hours-per-week']
sns.pairplot(train_set[cols], size = 2.5)
plt.show();
####dummy   for labels
joint = pd.concat([train_set, test_set], axis=0)
cat= joint.select_dtypes(include=['object']).axes[1]
for col in cat:
    joint = pd.concat([joint, pd.get_dummies(joint[col], prefix=col, prefix_sep=':')], axis=1)
    joint.drop(col, axis=1, inplace=True)
train_dum = joint.head(train_set.shape[0])
test_dum= joint.tail(test_set.shape[0])
####standardize the numerical features
scaler = StandardScaler()
scaler.fit(train_dum)
train_dum = scaler.transform(train_dum)
test_dum = scaler.transform(test_dum)
'''
data preprocessing and featrue engineering finished
firstly we use logistic regression \ decision tree
'''
print('=====build models=====')

'''
test model validation acc
'''
'''
search the best param:
    X_train, X_test, y_train, y_test = train_test_split(train_dum, train_label['income'], test_size=0.2, random_state=0)
    
    xg1 = XGBClassifier(learning_rate = 0.1, max_depth = 3, n_estimators = 250)
    param_grid = {'gamma':[0,10,50]}
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(xg1, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train,y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
then  we get the param.
'''
X_train, X_test, y_train, y_test = train_test_split(train_dum, train_label['income'], test_size=0.2, random_state=0)
xg2 = xgb.XGBClassifier(learning_rate=0.08,n_estimators=311,max_depth=6,min_child_weight=1)
model2=xg2.fit(X_train,y_train)
xgb.plot_importance(model2)
plt.show()
p2 = xg2.predict(X_train)
acc2=accuracy_score(y_train, p2)###accuracy of training set
p22 = model2.predict(X_test)
acc22=accuracy_score(y_test, p22)
confusion_matrix=confusion_matrix(y_test, p22)
#####model   xgboost
xg = xgb.XGBClassifier(learning_rate=0.08,n_estimators=311,max_depth=6,min_child_weight=1)
model1=xg.fit(train_dum,train_label['income'])
xgb.plot_importance(model1),plt.show()
predictions = xg.predict(train_dum)
acc=accuracy_score(train_label['income'], predictions)###accuracy of training set
predictions2 = model1.predict(test_dum)

print('=====write prediction=====')
def data_write_csv(filename, datas):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        for data in datas:
            writer.writerow([data])
    print('save file successfully')
    
data_write_csv('A2_pzhangan_20560903_prediction.csv', predictions2)
