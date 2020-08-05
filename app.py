import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,plot_precision_recall_curve
from sklearn.metrics import precision_score,recall_score

def main():
    st.title("Diabetes Classification Using ML Algorithms")
    st.sidebar.title("Classification")
    st.markdown("Do You Have Diabetes Or not?")
    st.sidebar.markdown("Lets Check It Out!!")

    @st.cache(persist=True)
    def load_data():
        data=pd.read_csv("diabetes.csv",encoding="latin")
        data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
        data['Glucose'].fillna(data['Glucose'].mean(),inplace=True)
        data['BloodPressure'].fillna(data['BloodPressure'].mean(),inplace=True)
        data['SkinThickness'].fillna(data['SkinThickness'].mean(),inplace=True)
        data['Insulin'].fillna(data['Insulin'].mean(),inplace=True)
        data['BMI'].fillna(data['BMI'].mean(),inplace=True)

        return data

    df=load_data()

    @st.cache(persist=True)
    def split(df):
        x=df.drop(columns="Outcome")
        y=df['Outcome']
        scaler=StandardScaler()
        standard_data=scaler.fit_transform(x)
        x_train,x_test,y_train,y_test=train_test_split(standard_data,y,test_size=0.3,random_state=2)
        return x_train,x_test,y_train,y_test

    class_names=['Diabetic','Not Diabetic']

    def plot_metrics(metris_list):
        if "Confusion Matrix" in metris_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model,x_test,y_test,display_labels=class_names)
            st.pyplot()
            
        if "ROC Curve" in metris_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model,x_test,y_test)
            st.pyplot()

        if "Precision Recall Curve" in metris_list:
            st.subheader("Precision Recall Curve")
            plot_precision_recall_curve(model,x_test,y_test)
            st.pyplot()
    
    x_train,x_test,y_train,y_test=split(df)

    if st.sidebar.checkbox("Show Raw data",False):
        st.subheader("Diabetes Dataset")
        st.write(df)
    
    st.sidebar.subheader("Choose Classifier")
    classifier=st.sidebar.selectbox("Classifier",("Support Vector Machine","Logistic Regression","Decision Tree","Random Forest Classifier","KNeighbors Classifier"))

    if classifier=="Support Vector Machine":
        st.sidebar.subheader("Model Parameters")
        c=st.sidebar.number_input("Regularization Parameter",0.01,10.0,step=0.01,key="c")
        kernel=st.sidebar.radio("kernel",("rbf","linear"),key="kernel")
        gamma=st.sidebar.radio("Gamma (Kernel Coefficient)",("scale","auto"),key="gamma")
        metrics=st.sidebar.multiselect("What Metrics to Plot?",("Confusion Matrix","ROC Curve","Precision Recall Curve"))
        if st.sidebar.button("Classify",key="classify"):
            st.subheader("Support Vector Machine Result")
            model=SVC(C=c,kernel=kernel,gamma=gamma)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy",accuracy.round(2))
            st.write("Precision",precision_score(y_test,y_pred).round(2))
            st.write("Recall",recall_score(y_test,y_pred).round(2))
            plot_metrics(metrics)

    if classifier=="Logistic Regression":
        st.sidebar.subheader("Model Parameters")
        c_lr=st.sidebar.number_input("Regularization Parameter",0.01,10.0,step=0.01,key="C_LR")
        max_iter=st.sidebar.slider("Maximum Number of Iterations",100,500,key="max_iter")
        metrics=st.sidebar.multiselect("What Metrics to Plot?",("Confusion Matrix","ROC Curve","Precision Recall Curve"))
        if st.sidebar.button("Classify",key="classify"):
            st.subheader("Logistic Regression Result")
            model=LogisticRegression(C=c_lr,max_iter=max_iter)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy",accuracy.round(2))
            st.write("Precision",precision_score(y_test,y_pred).round(2))
            st.write("Recall",recall_score(y_test,y_pred).round(2))
            plot_metrics(metrics)            
    
    if classifier=="Random Forest Classifier":
        st.sidebar.subheader("Model Parameters")
        n_estimators=st.sidebar.slider("The Number of Trees in the Forest",100,500,step=10,key="n_estimators")
        max_depth=st.sidebar.number_input("The Maximum Depth of the Tree",1,20,step=2,key="depth")
        bootstrap=st.sidebar.radio("Bootstrap samples when Building Trees",("True","False"),key="strap")
        criterion=st.sidebar.radio("Function to Measure the Quality of the Split",("gini","entropy"),key="criterion")
        metrics=st.sidebar.multiselect("What Metrics to Plot?",("Confusion Matrix","ROC Curve","Precision Recall Curve"))
        if st.sidebar.button("Classify",key="classify"):
            st.subheader("Random Forest Classifier Result")
            model=RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy",accuracy.round(2))
            st.write("Precision",precision_score(y_test,y_pred).round(2))
            st.write("Recall",recall_score(y_test,y_pred).round(2))
            plot_metrics(metrics)
    
    if classifier=="Decision Tree":
        st.sidebar.subheader("Model Parameters")
        max_depth_dt=st.sidebar.number_input("The Maximum Depth of the Tree",1,20,step=2,key="depth_dt")
        criterion_dt=st.sidebar.radio("Function to Measure the Quality of the Split",("gini","entropy"),key="criterion_dt")
        splitter=st.sidebar.radio("The Strategy used To Choose the Split at Each Node",("best","random"),key="splitter")
        min_sample_split=st.sidebar.slider("The minimum number of samples required to split an internal node",20,100,key="sample_split")
        metrics=st.sidebar.multiselect("What Metrics to Plot?",("Confusion Matrix","ROC Curve","Precision Recall Curve"))
        if st.sidebar.button("Classify",key="classify"):
            st.subheader("Decision Tree Classifier Result")
            model=DecisionTreeClassifier(criterion=criterion_dt,max_depth=max_depth_dt,splitter=splitter,min_samples_split=min_sample_split)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy",accuracy.round(2))
            st.write("Precision",precision_score(y_test,y_pred).round(2))
            st.write("Recall",recall_score(y_test,y_pred).round(2))
            plot_metrics(metrics)   
    
    if classifier=="KNeighbors Classifier":
        st.sidebar.subheader("Model Parameters")
        n_neighbors=st.sidebar.number_input("The Number of Neighbours to use",3,15,step=1,key="neighbors")
        metrics=st.sidebar.multiselect("What Metrics to Plot?",("Confusion Matrix","ROC Curve","Precision Recall Curve"))
        if st.sidebar.button("Classify",key="classify"):
            st.subheader("KNeighbors Classifier Result")
            model=KNeighborsClassifier(n_neighbors=n_neighbors,p=2,metric="euclidean")
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy",accuracy.round(2))
            st.write("Precision",precision_score(y_test,y_pred).round(2))
            st.write("Recall",recall_score(y_test,y_pred).round(2))
            plot_metrics(metrics) 
    
if __name__=='__main__':
    main()