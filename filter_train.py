"""
Feature select Algo based on graph approach:
x_train=  x1    x2       x3      x4
        [[a1     a2        a3      a4]
          [b1       b2      b3      b4]
          [c1       c2      c3      c4]]
create a graph with edges
(xi,xj) with edge weight distance b xi vector and xj vector
i,j=[1,4].
threshold= mean of weight of all edges
remove edges <threshold
calculate degree of each node remove the node with max degree from graph continue until null graph
null graph is graph with no edge ie weight of each edge is 0

the removed nodes(with max degree each time) are the selected features

for different threshold find diff features in term make new dataset
Train the new Dataset with Random Forest Classifier Model find accuracy

plot accuracy vs threshold 
select features whose threshold gives maximum accuracy
"""

############################ loading library ####################
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

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

#####################################loading dataset###################
path=input("enter dataset.csv name ")
f_name=input("enter name of file.txt where u want to store results ")
arr=np.loadtxt(path,delimiter=",",dtype=str)
_,n=arr.shape
x_train=arr[1:,:n-1]# slicing the dataset to remove column name(1st row) and the categorical column(last column)
x_train = x_train.astype(np.float64)#convert str to float
y = arr[1:,-1]#target column
_,n=x_train.shape

########################################################################

def wt_graph():
    graph=np.zeros([n, n], dtype = int)#store edge weight as dist between 2 features
    for i in range(n):
        for j in range(n):
            a=x_train[:,i]
            b=x_train[:,j]
            dist = np.linalg.norm(a-b)
            graph[i][j]=dist
    return graph

def f_select(threshold,graph,l):#returns the new dataset with selected features
    ###################### create edge graph ##############################
    graph1=np.ones([n, n], dtype = int)# store edge present or absent(1/0)

    while(graph1.sum()!=0):# check for null graph ie no edge present ie edge weight 0
        graph1=np.zeros([n, n], dtype = int)# store edge present or absent
        for i in range(n):
            for j in range(n):
                if(graph[i][j]>threshold):
                    graph1[i][j]=1
        #print(graph1)
        ############################ find degree of each node ##################
        degree={}
        cnt=0
        for i in range(n):
            for j in range(n):
                if(graph1[i][j]==1):
                    cnt=cnt+1
            degree[i]=cnt
            cnt=0
        #####################################find node with max degree ##############
        Keymax = max(zip(degree.values(), degree.keys()))[1]
        l.append(Keymax)            
        ########################################remove node with max degree###############
        graph[:,Keymax]=0# set col to 0
        graph[Keymax]=0#set row to 0
        graph1[:,Keymax]=0# set col to 0
        graph1[Keymax]=0#set row to 0

    l=sorted(l)
    return (x_train[:,l],l)

def loop(u):#u is the update value of threshold
    graph=wt_graph()
    temp=graph
    temp=np.unique(np.sort(temp.flatten()))
    lower=0#temp[1]#when used as threshold give max features
    upper=temp[-2]# when used as threshold give one feature
    with open(f_name, 'a') as file:

        lp=[]
        la=[]
        for i in range(lower,upper,u):
            temp=wt_graph()
            (x_new,l)=f_select(i,temp,[])
        
            ###################################  Train the model to find precision #################
            X_train, X_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2, random_state=42)
            accuracy=precision=0

            for name, clf in classifiers.items():
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy =accuracy+ accuracy_score(y_test, y_pred)
                precision = precision+ precision_score(y_test, y_pred, average='weighted')
                
            accuracy=accuracy/11
            precision=precision/11
            la.append((accuracy,i,l))
            lp.append((precision,i,l))
            #writing results in text file
            file.write(f"precision:{precision:0.5f} acuracy:{accuracy:0.5f} with threshold:{i} with features selected:{l} \n")
    return (lp,la)

############################## user interface ##########################
g=wt_graph()
t=g
t=np.unique(np.sort(t.flatten()))
th=np.mean(g)

with open(f_name, 'w') as file:
    file.write(f"{path} dataset with {arr.shape} size\n")
    file.write(f"avg weight edge threshold:{th} lower limit :{0} upper limit :{t[-2]}\n")

print(f"avg weight edge threshold:{th} lower limit :{0} upper limit :{t[-2]}")
u=int(input("enter update constant to loop from lower to upper threshold "))
lp,la=loop(u)
lp = sorted(lp, key=lambda x: x[0], reverse=True)
la = sorted(la, key=lambda x: x[0], reverse=True)
list1=lp[0][2]
list2=la[0][2]
intersection = list(set(list1).intersection(list2))

with open(f_name, 'a') as file:
    file.write(f"{lp[0][2]} is selected feature columns with maximum precission:{lp[0][0]:0.2f}\n")
    file.write(f"{la[0][2]} is selected feature columns with maximum accuracy:{la[0][0]:0.2f}\n")
    file.write(f"features with both max accuracy and precission :{intersection}\n")

x_main=x_train[:,intersection]
np.save("Trimdata.npy",x_main)

