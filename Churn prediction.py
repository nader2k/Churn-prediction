
#churn prediction
#importing the librarys for churn prediction
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,recall_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
#loading the excel file of the company
df = pd.read_csv("f:/retailchurners.csv", engine='python',encoding='latin1')
df.head()
df.dtypes
df.shape
df.describe()
#get the specific dates of customers purchases
df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate)
df['date'] = pd.to_datetime(df.InvoiceDate.dt.date)
df['time'] = df.InvoiceDate.dt.time
df['hour'] = df['time'].apply(lambda x: x.hour)
df['weekend'] = df['date'].apply(lambda x: x.weekday() in [5, 6])
df['dayofweek'] = df['date'].apply(lambda x: x.dayofweek)
df['week'] = df['date'].apply(lambda x: x.week)
df['month'] = df['date'].apply(lambda x: x.month)

#find the histogram 
pd.plotting.scatter_matrix(df, hist_kwds={'bins':15},figsize=(10,10),color='darkblue')
fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
sns.boxplot(y='creditscore',x = 'exited', hue = 'exited',data = df, ax=axarr[0][0])
sns.boxplot(y='age',x = 'exited', hue = 'exited',data = df , ax=axarr[0][1])
sns.boxplot(y='tenure',x = 'exited', hue = 'exited',data = df, ax=axarr[1][0])
sns.boxplot(y='balance',x = 'exited', hue = 'exited',data = df, ax=axarr[1][1])
sns.boxplot(y='numofproducts',x = 'exited', hue = 'exited',data = df, ax=axarr[2][0])
sns.boxplot(y='estimatedsalary',x = 'exited', hue = 'exited',data = df, ax=axarr[2][1])
#find outliers
def outlier_thresholds(dataframe, variable, low_quantile=0.05, up_quantile=0.95):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit
def has_outliers(dataframe, numeric_columns, plot=False):
   # variable_names = []
    for col in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, " : ", number_of_outliers, "outliers")
            #variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()

for var in numeric_variables:
    print(var, "has " , has_outliers(df, [var]),  "Outliers")
#create X data for featurs that we want to use in our models (independent variables)
X = df.drop(['churn','weekmedian','avg_order_value','returns','customerid','month','week','dayofweek','weekend','hour','time','date','InvoiceDate'])
y = df['churn']
#split the data in to train and test we choose 20 percent of data for testing beacaue we want to evaluate the model for each 2 years of data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
def models():
    """this function train ensemble learning models on the train data 
    and then it will test models on the test data"""
    knn = KNeighborsClassifier(random_state=0).fit(X_train, y_train)
    #n_estimators means the number of decision trees use for ensemble learning
    forest = RandomForestClassifier(n_estimators=200)
    forest.fit(X_train,y_train)
    GBC=GradientBoostingClassifier(max_depth=8).fit(X_train, y_train)
    #Optimize max_depth parameter. It represents the depth of each tree, which is the maximum number of different features used in each tree a high max_depth you can come up with overfiting
    xgb = XGBClassifier(n_estimators=100,max_depth=3)
    xgb.fit(X_train,y_train)
    #(num_leaves) This is the main parameter to control the complexity of the tree model. Theoretically, we can set num_leaves = 2^(max_depth) to obtain the same number of leaves as depth-wise tree,
    #we can use this parameter only for lightgbm model
    lgbm = LGBMClassifier(n_estimators=120,max_depth=5,num_leaves=50)
    lgbm.fit(X_train,y_train)
    #Print the accuracy for each model
    print("random forest",classification_report(forest.predict(X_test),y_test))
    print("XG-Boost",classification_report(xgb.predict(X_test),y_test))
    print("LightGBM",classification_report(lgbm.predict(X_test),y_test))
    print("KNN",classification_report(knn.predict(X_test),y_test))
    print("GBC",classification_report(GBC.predict(X_test),y_test))

    return forest,xgb,lgbm,knn,GBC

forest,xgb,lgbm,knn,GBC = models()
# Auc Roc Curve
def generate_auc_roc_curve(clf, X_test):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="AUC ROC Curve with Area Under the curve ="+str(auc))
    plt.legend(loc=4)
    plt.show()
    pass
generate_auc_roc_curve(forest, X_test)
generate_auc_roc_curve(xgb, X_test)
generate_auc_roc_curve(lgbm, X_test)
generate_auc_roc_curve(knn, X_test)
generate_auc_roc_curve(GBC, X_test)
Importance = pd.DataFrame({"Importance": lgbm.feature_importances_*100},
                         index = X.columns)

Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "purple")

plt.xlabel("Variable Severity Levels")