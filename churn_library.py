"""
This is a churn_library.py python file that we used to find cystomers who
are likely to churn. The execution of this file wille produce aetefacts in images and models folders.
Date: April 02,2023
Author: Hermione ODJO
"""

#Libraries import
import os
import logging

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import joblib

#Log file
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO
    filemode='w',
    format='%(name)s - %(levelname)s -%(message)s'
)

#Data imports


def import_data(pth):
    '''
    returns dataframe for the csv found at pth
    input: 
        pth : a path to the csv
        
    output: 
        df: pandas dataframe
    '''
    df=pd.read_csv(pth)
    df.drop('customerID',axis=1,inplace=True)
    df['Churn']=df['Churn'].apply(lambda val: 0 if val=='No' else 1)

    return df

def data_spliting(df):
    """
    input :
        df: pandas dataframe
    outputs:
        train :train dataframe
        validate: validation dataframe
        test test dataframe
    """
    train,test=train_test_split(df,test_size=0.3,random_state=seed,stratify=df['Churn'])
    test,validate=train_test_split(test,test_size=0.5,random_state=seed,stratify=test['Churn'])
    #Saving the different data sets
    train.to_csv('./data/train.csv',index=False)
    test.to_csv('./data/test.csv',index=False)
    validate.to_csv('./data/validate.csv',index=False)
    
    X_train=train.drop('Churn',axis=1)
    X_val=validate.drop('Churn',axis=1)

    y_train=train['Churn']
    y_val=validate['Churn']

    return train, X_train,y_train,X_val,y_val
    
def performs_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
        df_paths: a pandas dataframe
        
    output:
        None
    
    """
    df_copy=df.copy()
    list_colums=df_copy.columns.to_list()
    list_colums.append('Heatmap')
    
    df_corr=df_copy.corr(numeric_only=True)
    
    for column_name in list_colums:
        plt.figure(figsize=(10,6))
        if column_name=='Heatmap':
            sns.heatmap(
                df_corr,
                mask=np.triu(np.ones_like(df_corr,dtype=bool)),
                center=0,cmap='RdBu', linewidths=1,annot=True,
                fmt=".2f",vmin=-1, vmax=1
            )
        else:
            if df[column_name].dtype !='O':
                df.column_name.hist()
            else :
                sns.countplot(data=df, x=column_name)
        plt.savefig('images/eda/'+column_name+'.jpg')
        plt.close()
        

def classification_report_image(y_train,
                                y_train_pred,
                                y_val,
                                y_val_pred):
    """
    produces classification report for training and testing results and stores report
    as image in image folder
    
    inputs:
            y_train: training dataset real  labels
            y_train_pred: labels predicted from training dataset with logistic regression
            y_val: validation dataset real  labels
            y_train_pred: labels predicted from validation dataset with logistic regression
    outputs :
            None

    """
    class_report_dict={
        "Logistic Regression train results" : classification_report(y_train,y_train_pred),
        "Logistic Regression validation results" : classification_report(y_val,y_val_pred)

    }
    
    for title, report in class_report_dict.items():
        plt.rc('figure',figsize=(7,3))
        plt.text( 0.2, 0.3, str(report), {'fontsize':10 }, fontproperties='monospace');
        plt.axis('off')
        plt.title(title,fontweight='bold'),
        plt.save('images/results/'+title+'.jpg')
        


def train_model():
    
    return

def build_pipline():
    """
    Convert TotalChargeColumns to numeric, normalize numeric columns,  encode 
    categoricals columns, and set 
    """
    numeric_features=['SeniorCitizen', 'tenure', 'MonthlyCharges','TotalCharges']

    categorical_features= ['gender',
                        'Partner',
                        'Dependents',
                        'PhoneService',
                        'MultipleLines',
                        'InternetService',
                        'OnlineSecurity',
                        'OnlineBackup',
                        'DeviceProtection',
                        'TechSupport',
                        'StreamingTV',
                        'StreamingMovies',
                        'Contract',
                        'PaperlessBilling',
                        'PaymentMethod',
                        ]
    #Pipeline for numeric variables processing
    numeric_transformer=Pipeline(
        steps=[('convert',FunctionTransformer(convert_totalcharges)),
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())]
    )

    #Pipeline categorical variables processing
    categorical_transformer=Pipeline(
        steps=[('Onehotencoder',OneHotEncoder(sparse=False,handle_unknown='ignore'))]
    )

    #Data prepocessing pipeline
    preprocessor=ColumnTransformer(
        transformers=[('numeric',numeric_transformer,numeric_features),
                    ('categorical',categorical_transformer,categorical_features)]
    )

    #Data modélisation pipeline with random forest algorith

    pipe_model=Pipeline(
        steps=[('preprocessor',preprocessor),
            ('logreg',LogisticRegression(random_state=123,
                                         solver='newton-cg',
                                         max_iter=2000,
                                         c=5.0,
                                         penalty='l2'))]
    )
    
    return pipe_model

def 