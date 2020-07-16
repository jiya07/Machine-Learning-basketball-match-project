import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score

df = pd.read_csv('cbb.csv')
print(df.head())

#Adding a column called 'windex' that will contain value as 'True' if the wins above bubble(WAB) are over 7 and 'False' if not. 
df['windex'] = np.where(df.WAB > 7, 'True', 'False')

#Filtering the dataset by keeping the teams that made to 'Sweet sixteen','the Elite Eight', and 'the Final Four'
df1 = df.loc[df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
print(df1['POSTSEASON'].value_counts())

#Coverting the categorical values to numerical values
df1.groupby(['windex'])['POSTSEASON'].value_counts(normalize=True)
df1['windex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)

#Feature set
X = df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D','TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O','3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
#Standardisation of data to give 0 mean and unit variance
X= preprocessing.StandardScaler().fit(X).transform(X)
# X[0:5]

#Target column
y = df1['POSTSEASON'].values

#Spillting data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Validation set:', X_val.shape,  y_val.shape)

#Traing with KNN model and then checking the accuracy on validation set
#Finding the right value of k
Ks = 16
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1,Ks):  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_val)
    mean_acc[n-1] = accuracy_score(y_val, yhat)
    std_acc[n-1]=np.std(yhat==y_val)/np.sqrt(yhat.shape[0])
print(mean_acc)
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
n=KNeighborsClassifier(5).fit(X_train,y_train)
knn_yhat=n.predict(X_val)
print("\nK nearest neighbours")
print("Accuracy : ",accuracy_score(y_val,knn_yhat))

#Decision tree model
decTree=DecisionTreeClassifier(criterion='entropy',max_depth=1)
decTree.fit(X_train,y_train)
predTree=decTree.predict(X_val)
print(predTree[0:10])
print(y_val[0:10])
print("\nDecsion Tree")
print("Accuracy: ", accuracy_score(y_val, predTree))

#SVM Model
clf = svm.SVC(kernel='poly',gamma='auto').fit(X_train, y_train) 
svm_yhat = clf.predict(X_val)
# set(y_val) - set(svm_yhat)
print("\nSupport Vector Machine")
print("Accuracy: ", accuracy_score(y_val, svm_yhat))

#Logistic Regression
lr= LogisticRegression(C=0.01, solver='lbfgs').fit(X_train,y_train)
lr_yhat = lr.predict(X_val)
print(yhat)
lr_yhat_prob = lr.predict_proba(X_val)
print(lr_yhat_prob[0])
print("\nLogistic Regression")
print("Accuracy: ",accuracy_score(y_val,lr_yhat))


#EVALUATION OF MODEL USING TEST SET
test_df = pd.read_csv('basketball_train.csv',error_bad_lines=False)
# test_df.head()

test_df['windex'] = np.where(test_df.WAB > 7, 'True', 'False')
test_df1 = test_df[test_df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
test_Feature = test_df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D','TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O','3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
test_Feature['windex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)
test_X=test_Feature
test_X= preprocessing.StandardScaler().fit(test_X).transform(test_X)
# test_X[0:3]
test_y = test_df1['POSTSEASON'].values
    
def jaccard_index(predictions, true):
    if (len(predictions) == len(true)):
        intersect = 0;
        for x,y in zip(predictions, true):
            if (x == y):
                intersect += 1
        return intersect / (len(predictions) + len(true) - intersect)
    else:
        return -1

#KNN MODEL
K = 10
mean_acc = np.zeros((K-1))
std_acc = np.zeros((K-1))
for n in range(1,K):
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(test_X)
    mean_acc[n-1] = accuracy_score(test_y, yhat)
    std_acc[n-1]=np.std(yhat==test_y)/np.sqrt(yhat.shape[0])
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
neighb = KNeighborsClassifier(1).fit(X_train,y_train)
knn_yhat=neighb.predict(test_X)
print("\nK nearest neighbours")
print("Accuracy: ", accuracy_score(test_y,knn_yhat))
print("Jaccard index: ",jaccard_index(knn_yhat,test_y))
print("F1 score :",f1_score(test_y,knn_yhat,average='micro'))

#DECISION TREE
decTree=DecisionTreeClassifier(criterion='entropy',max_depth=9)
decTree.fit(X_train,y_train)
predTree=decTree.predict(test_X)
print("\nDecsion Tree")
print("Accuracy: ", accuracy_score(test_y, predTree))
print("Jaccard index: ",jaccard_index(predTree,test_y))
print("F1 score :",f1_score(test_y,predTree,average='micro'))

#SVM
clf = svm.SVC(kernel='linear',gamma='auto').fit(X_train, y_train) 
svm_yhat = clf.predict(test_X)
# set(y_val) - set(yhat)
print("\nSupport Vector Machine")
print("Accuracy: ", accuracy_score(test_y, svm_yhat))
print("Jaccard index: ",jaccard_index(svm_yhat,test_y))
print("F1 score :",f1_score(test_y, svm_yhat,average='micro'))

#LOGISTIC REGRESSION
lr= LogisticRegression(C=0.12, solver='lbfgs',multi_class='auto').fit(X_train,y_train)
lr_yhat = lr.predict(test_X)
lr_yhat_prob = lr.predict_proba(test_X)
print("\nLogistic Regression ")
print("Accuracy: ", accuracy_score(test_y, lr_yhat))
print("Jaccard index: ",jaccard_index(lr_yhat,test_y))
print("F1 score :",f1_score(test_y, lr_yhat,average='micro'))
print("Log Loss: ",log_loss(test_y, lr_yhat_prob))

