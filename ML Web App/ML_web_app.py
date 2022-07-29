import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px 

## Heading App 
st.write(""" ## **Explore different ML models and Datasets** """)


# datasets ka Selector
dataset_name=st.sidebar.selectbox("Select Dataset",["Iris","Breast Cancer","Wine","Diabetes"])

# Classifiers Ka Box 
classifier_name=st.sidebar.selectbox("Select Classifier",["KNN","SVM","Random Forest"])

# creating a function for Load Dataset
def load_dataset(dataset_name):
    data=None
    if dataset_name=="Iris":
        data=datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        data=datasets.load_breast_cancer()
    elif dataset_name=="Wine":
        data=datasets.load_wine()
    elif dataset_name=="Diabetes":
        data=datasets.load_diabetes()
    x=data.data
    y=data.target
    return x,y

x,y=load_dataset(dataset_name)

# shape of Datasets 
st.write("Shape of Dataset:",x.shape)
st.write("Number of Classes : ",len(np.unique(y)))

# Parameters of Different Classifiers 

def add_parameter(classifier_name):
    params=dict()
    if classifier_name=="SVM":
        C=st.sidebar.slider("C",0.01,10.0)
        params["C"]=C
    elif classifier_name=="KNN":
        K=st.sidebar.slider("K",1,15)
        params["K"]=K
    elif classifier_name=="Random Forest":
        max_depth=st.sidebar.slider("Max Depth",2,15)
        params['max_depth']=max_depth
        n_estimators=st.sidebar.slider("Number of Estimators",1,100)
        params['n_estimators']=n_estimators
    return params
params=add_parameter(classifier_name)

def get_classifier(classifier_name,parmas):
    clf=None
    if classifier_name=="SVM":
        clf=SVC(C=params["C"])
    elif classifier_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier_name=="Random Forest":
        clf=RandomForestClassifier(max_depth=params['max_depth'],n_estimators=params['n_estimators'])
    return clf
clf=get_classifier(classifier_name,params)

train_size=st.sidebar.selectbox("Train Size",[0.7,0.8,0.9])
random_state=st.sidebar.selectbox("Random State",[0,1,21,33,42])
X_train,X_test,y_train,y_test=train_test_split(x,y,train_size=train_size,random_state=random_state)

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
st.write(f'Accuracy of {classifier_name} on {dataset_name} dataset is {accuracy}')

pca=PCA(2)
X_projected=pca.fit_transform(x)

x1=X_projected[:,0]
x2=X_projected[:,1]

fig=plt.figure()
plt.scatter(x1,x2,c=y,s=100,cmap='viridis',animated=True)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'{dataset_name} Dataset')
plt.colorbar()
st.pyplot(fig)










