# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:52:16 2020

@author: Rajneesh
"""
#Impute data for feature "LoanAmount"
def imputena(data):
    var=data['Income_var']
    
    if(var=='A'):
        return 133.58
    elif(var=='B'):
        return 249.47
    elif(var=='C'):
        return 495
    elif(var=='D'):
        return 283
    elif(var=='E'):
        return 700
    elif(var=='F'):
        return 490
    else:
        return 360
    
#impute data for feature "Loan_Amount_Term"    
def impute_term(data):
    var=data['amount_var']
    
    if(var=='A'):
        return 341.08
    elif(var=='B'):
        return 349.41
    elif(var=='C'):
        return 332.30
    else:
        return 390
    
    
#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Reading train and test data
df_train=pd.read_csv('train_loan.csv')
df_test=pd.read_csv('test_lAUu6dG.csv')

#separating the loan ID column as we need it in the output file for submission
Loan_IDs=df_test['Loan_ID']

#Exploring the data
print(df_train.head())

print(df_train.describe())

print(df_train.shape)

print(df_train.corr())


#Checking the number of missing values in the data
sns.heatmap(df_train.isnull(),cbar=False,yticklabels=False)
plt.show()

sns.countplot(df_train['Dependents'])
#changing the int 0 with string 0 as it it mistyped 
df_train['Dependents'].replace(0,'0',inplace=True)
df_test['Dependents'].replace(0,'0',inplace=True)


#Filling NAs for Gender feature
print(df_train['Gender'].value_counts())
#Filling the NA by mode of Gender feature 
sns.countplot(df_train['Gender'])
plt.show()

'''
df_train['Gender'].fillna('Male',inplace=True)
df_test['Gender'].fillna('Male',inplace=True)
'''

#Filling the NAs for Married feature
sns.countplot(df_train['Dependents'])
plt.show()
#Filling as per larger value counts(mode of the variable)
df_train['Married'].fillna('Yes',inplace=True)
df_test['Married'].fillna('Yes',inplace=True)


#Filling the NAs for Dependents column by calculating the mode
sns.countplot(df_train['Dependents'])
plt.show()

df_train['Dependents'].fillna(0,inplace=True)
df_test['Dependents'].fillna(0,inplace=True)

#filling the NAs for Self employed by calculating the mode
df_train['Self_Employed'].fillna('No',inplace=True)
df_test['Self_Employed'].fillna('No',inplace=True)

#creating a new feature Income_var from the Applicant Income column. It will help us categorize the Income of applicants, by which we can fill the NAs of LoanAmount feature
#As LoanAmount is highly correlated to Applicant Income, We are creating this column
bins=[0,10000,20000,30000,40000,50000,60000,70000,80000,120000]
labels=['A','B','C','D','E','F','G','H','I']

loan_cut_train=pd.cut(df_train['ApplicantIncome'],bins=bins,labels=labels)
loan_cut_test=pd.cut(df_test['ApplicantIncome'],bins=bins,labels=labels)

income_category_train=pd.concat([df_train['LoanAmount'],loan_cut_train],axis=1)
income_category_test=pd.concat([df_test['LoanAmount'],loan_cut_test],axis=1)

income_category_train.rename(columns={'ApplicantIncome':'Income_var'},inplace=True)
income_category_test.rename(columns={'ApplicantIncome':'Income_var'},inplace=True)

income_pivot_train=income_category_train.pivot_table('LoanAmount','Income_var')
income_pivot_test=income_category_test.pivot_table('LoanAmount','Income_var')

print(income_pivot_train)
#Adding the Income_var column in the main dataframe.
df_train=pd.concat([income_category_train['Income_var'],df_train],axis=1)
df_test=pd.concat([income_category_test['Income_var'],df_test],axis=1)

df_test['Income_var'].fillna('A',inplace=True)
#filling the missing values in the LoanAmount column
df_train['LoanAmount']=df_train.apply(lambda x:imputena(x) if pd.isnull(x['LoanAmount']) else x['LoanAmount'],axis=1)
df_test['LoanAmount']=df_test.apply(lambda x:imputena(x) if pd.isnull(x['LoanAmount']) else x['LoanAmount'],axis=1)

#finding relation between Loan_Amount_Term and other columns so that we can fill missing values accordingly
print(df_train.corr())
#As term of Loan Amount will be dependent on Loan Amount, finding relation between that.
#A loan with high amount will have a long term as compared to less amount loan
plt.scatter(df_train['Loan_Amount_Term'],df_train['LoanAmount'])
plt.show()
#creating bins and labels to categorize the LoanAmount column
bins1=[0,200,400,600,800]
labels1=['A','B','C','D']

loanamount_cut_train=pd.cut(df_train['LoanAmount'],bins=bins1,labels=labels1)
loanamount_cut_test=pd.cut(df_test['LoanAmount'],bins=bins1,labels=labels1)

print(loanamount_cut_train)

amount_cut_train=pd.DataFrame(data=loanamount_cut_train)
amount_cut_test=pd.DataFrame(data=loanamount_cut_test)

amount_cut_train.rename(columns={'LoanAmount':'amount_var'},inplace=True)
amount_cut_test.rename(columns={'LoanAmount':'amount_var'},inplace=True)


df_train=pd.concat([df_train,amount_cut_train],axis=1)
df_test=pd.concat([df_test,amount_cut_test],axis=1)

#Filling the missing values in the Loan_Amount_Term 
df_train['Loan_Amount_Term']=df_train.apply(lambda x:impute_term(x) if pd.isnull(x['Loan_Amount_Term']) else x['Loan_Amount_Term'],axis=1)
df_test['Loan_Amount_Term']=df_test.apply(lambda x:impute_term(x) if pd.isnull(x['Loan_Amount_Term']) else x['Loan_Amount_Term'],axis=1)

#filling the missing values in Credit_history column by taking mode as it's not dependent on any other feature
sns.countplot(df_train['Credit_History'])
plt.show()

df_train['Credit_History'].fillna(1.0,inplace=True)
df_test['Credit_History'].fillna(1.0,inplace=True)

#Changing the type Loan_Status colummn from object to int
Loan_Status=pd.get_dummies(df_train['Loan_Status'],drop_first=True)

df_train.drop('Loan_Status',inplace=True,axis=1)
df_train=pd.concat([df_train,Loan_Status],axis=1)

df_train.rename(columns={'Y':'Loan_Status'},inplace=True)

#creating dummies for the categorical variables
loan_dummy_train=pd.get_dummies(df_train[['Gender','Married','Education','Self_Employed','Property_Area']],drop_first=True)
loan_dummy_test=pd.get_dummies(df_test[['Gender','Married','Education','Self_Employed','Property_Area']],drop_first=True)

df_train.drop(['Gender','Married','Education','Self_Employed','Property_Area'],inplace=True,axis=1)
df_test.drop(['Gender','Married','Education','Self_Employed','Property_Area'],inplace=True,axis=1)

df_train=pd.concat([df_train,loan_dummy_train],axis=1)
df_test=pd.concat([df_test,loan_dummy_test],axis=1)

#Adding a new column Total_Income which is sum of Applicant Income and Coapplicant Income
df_train['Total_Income']=df_train['ApplicantIncome']+df_train['CoapplicantIncome']
df_test['Total_Income']=df_test['ApplicantIncome']+df_test['CoapplicantIncome']

#Label Encoding the Income_var and Dependents column 
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_train['Income_label']=le.fit_transform(df_train['Income_var'])
df_test['Income_label']=le.fit_transform(df_test['Income_var'])

df_train.drop(['Income_var','Loan_ID','amount_var'],axis=1,inplace=True)
df_test.drop(['Income_var','Loan_ID','amount_var'],axis=1,inplace=True)


df_train['Dependents']=df_train['Dependents'].apply(lambda x:str(x))
df_test['Dependents']=df_test['Dependents'].apply(lambda x:str(x))

df_train['Dependents_label']=le.fit_transform(df_train['Dependents'])
df_test['Dependents_label']=le.fit_transform(df_test['Dependents'])

df_train.drop('Dependents',axis=1,inplace=True)
df_test.drop('Dependents',axis=1,inplace=True)

df_train.drop('ApplicantIncome',axis=1,inplace=True)
df_test.drop('ApplicantIncome',axis=1,inplace=True)

'''
#checking the model with Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X,y)
predictions=lr.predict(df_test)
'''

#Scaling the dataframe
from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
normalized_data=mms.fit_transform(df_train)
norm_train=pd.DataFrame(normalized_data,columns=df_train.columns)

normalized_test=mms.fit_transform(df_test)
norm_test=pd.DataFrame(normalized_test,columns=df_test.columns)

X=norm_train.drop('Loan_Status',axis=1)
y=norm_train['Loan_Status']

#Predicting the output using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=289,
 min_samples_split=4,
 min_samples_leaf=9,
 max_features='auto',
 max_depth= 71,
 bootstrap='True')
rfc.fit(X,y)
predictions=rfc.predict(norm_test)

#checking the cross Validation score for RandomForestClassifier model
from sklearn.model_selection import cross_val_score
print(cross_val_score(rfc,X,y))


#creating a dataframe with Loan ID and predictions as columns
output = pd.DataFrame({'Loan_ID': Loan_IDs, 'Loan_Status': predictions})
output['Loan_Status'].replace({0:'N',1:'Y'},inplace=True)
output.to_csv('Loan Predictor Output.csv', index=False)
