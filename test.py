import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from filter_train import path,f_name
# Load the  dataset
x=np.load("Trimdata.npy")#feature dataset
#path=input("enter dataset.csv name ")
arr=np.loadtxt(path,delimiter=",",dtype=str)
y = arr[1:,-1]#target column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gaussian':GaussianProcessClassifier(),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
}

# Train each classifier and evaluate its performance on the test set
#f_name=input("enter name of file where u want to store precision vs threshold for diff classifiers ")
with open(f_name, 'a') as file:
    i=0
    for name, clf in classifiers.items():
        i=i+1
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average='weighted')
        file.write(f'{i} : {name} Classifier accuracy: {acc:.5f} precission :{pre:.5f}\n')

print("write in file finish \nU may now open it")



