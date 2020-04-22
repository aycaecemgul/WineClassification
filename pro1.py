import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#getting the data
dataset = pd.read_csv("wine.csv")

X=dataset.iloc[:,0:13]
y=dataset.iloc[:,13]

#DECISION TREE CLASSIFIER
print("Decision Tree")
#creating the decision tree classifier
dt_clf = DecisionTreeClassifier(random_state=0)
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0, stratify=y)
#teaching
dt_clf.fit(X_train_dt, y_train_dt)
#getting the predictions of algorithm
dt_result = dt_clf.predict(X_test_dt)
print("Accuracy score: ")
print(accuracy_score(dt_result, y_test_dt))
print("Confusion matrix: ")
print(confusion_matrix(y_test_dt, dt_result))
print("10-Fold Cross Validation Score:")
print(cross_val_score(dt_clf, X, y, cv=10))

print("")

#K-NN NEIGHBOUR
print("K-Nearest Neighbour")
sc_X=StandardScaler()
X_train_kn, X_test_kn, y_train_kn, y_test_kn = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0, stratify=y)
X_train_kn = sc_X.fit_transform(X_train_kn)
X_test_kn=sc_X.fit_transform(X_test_kn)
classifier=KNeighborsClassifier(n_neighbors=11,p=2,metric="euclidean")
classifier.fit(X_train_kn,y_train_kn)
kn_pred=classifier.predict(X_test_kn)
print("Accuracy Score:")
print(accuracy_score(kn_pred,y_test_kn))
print("Confusion matrix: ")
print(confusion_matrix(y_test_kn,kn_pred))
print("10-Fold Cross Validation Score:")
print(cross_val_score(classifier, X, y, cv=10))

input = [float(x) for x in input("Enter the data: ").split(",")]
new_prediction=np.array(input).reshape(1,-1)
new_result=dt_clf.predict(new_prediction)
print("Result:")
print(new_result)
