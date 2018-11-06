import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as f1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

#importing the dataset
data = pd.read_csv('zoo.data')
#Class labels
Y = np.ravel(np.column_stack(data["type"]).T)
score = []

#Class features
X = np.column_stack((data["hair"],data["feathers"],data["eggs"],data["milk"],data["airborne"],data["aquatic"],data["predator"],data["toothed"],data["backbone"],data["breathes"],data["venomous"],data["fins"],data["legs"],data["tail"],data["domestic"],data["catsize"]))

#splitting the dataset into training and test set
x_train,x_test,y_train,y_test = train_test_split(X,Y,shuffle=False)

#Decision tree
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_dt = dt.predict(x_test)
score.append(dt.score(x_test,y_test))
print('           Decision Tree Classifier         ')
print(classification_report(y_test,y_dt))
print(f1(y_test,y_dt,average='micro'))

#KNN
knn = KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto')
knn.fit(x_train,y_train)
y_knn = knn.predict(x_test)
score.append(knn.score(x_test,y_test))
print('              K-Nearest Neighbours          ')
print(classification_report(y_test,y_knn))
print(f1(y_test,y_knn,average='micro'))

#Logistic Regression
lr = LogisticRegression(penalty='l2',C=5,solver='lbfgs')
lr.fit(x_train,y_train)
y_lr = lr.predict(x_test)
score.append(lr.score(x_test,y_test))
print('                 Logistic Regression      ')
print(classification_report(y_test,y_lr))
print(f1(y_test,y_lr,average='micro'))

#LDA
lda = LinearDiscriminantAnalysis(solver='svd')
lda.fit(x_train,y_train)
y_lda = lda.predict(x_test)
score.append(lda.score(x_test,y_test))
print('                 Linear Discriminant Analysis         ')
print(classification_report(y_test,y_lda))
print(f1(y_test,y_lda,average='micro'))

#Perceptron
per = Perceptron(penalty='l1',alpha=0.001)
per.fit(x_train,y_train)
y_per = per.predict(x_test)
score.append(per.score(x_test,y_test))
print('                 Perceptron Classifier         ')
print(classification_report(y_test,y_per))
print(f1(y_test,y_per,average='micro'))

#SVM
svm = SVC(C=100,kernel='poly',degree=4)
svm.fit(x_train,y_train)
y_svm = svm.predict(x_test)
score.append(svm.score(x_test,y_test))
print('                Support Vector Machine           ')
print(classification_report(y_test,y_svm))
print(f1(y_test,y_svm,average='micro'))

#Naives Bayes Algorithm
nb = MultinomialNB(alpha=0.2)
nb.fit(x_train,y_train)
y_nb = nb.predict(x_test)
score.append(nb.score(x_test,y_test))
print('                Naives Bayes Algorithm            ')
print(classification_report(y_test,y_nb))
print(f1(y_test,y_nb,average='micro'))

#Comparison of above classification algorithms
n = np.arange(7)
plt.bar(n,score)
plt.xticks(n,('Decision Tree','kNN','Logistic','LDA','Perceptron','SVM','Naives Bayes'))
plt.xlabel('Classification Algorithm')
plt.ylabel('F1 score')
plt.title('Comparison of Classification Algorithms')
plt.show()

