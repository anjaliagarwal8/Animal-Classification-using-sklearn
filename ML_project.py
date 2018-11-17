import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier

#importing the dataset
data = pd.read_csv('zoo.data')
#Class labels
Y = np.ravel(np.column_stack(data["type"]).T)
score = []

#Class features
X = np.column_stack((data["hair"],data["feathers"],data["eggs"],data["milk"],data["airborne"],data["aquatic"],data["predator"],data["toothed"],data["backbone"],data["breathes"],data["venomous"],data["fins"],data["legs"],data["tail"],data["domestic"],data["catsize"]))

#Principal Component Analysis
Pca = PCA(n_components=3,svd_solver='full')
y = Pca.fit_transform(X)

#splitting the dataset into training and test set
x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=0,shuffle=False)
pca_trainx,pca_testx,pca_trainy,pca_testy = train_test_split(y,Y,shuffle=False)

#Finding the best model for the dataset
#Decision tree
dt = DecisionTreeClassifier(max_depth=6)
dt.fit(x_train,y_train)
y_dt = dt.predict(x_test)
score.append(dt.score(x_test,y_test))
print('           Decision Tree Classifier         ')
print(cross_val_score(dt,x_train,y_train,cv=3))
print(f1(y_test,y_dt,average='micro'))

#KNN
knn = KNeighborsClassifier(n_neighbors=10,weights='uniform',algorithm='auto')
knn.fit(x_train,y_train)
y_knn = knn.predict(x_test)
score.append(knn.score(x_test,y_test))
print('              K-Nearest Neighbours          ')
print(cross_val_score(knn,x_train,y_train,cv=3))
print(f1(y_test,y_knn,average='micro'))

#Logistic Regression
lr = LogisticRegression(penalty='l2',C=5,solver='lbfgs')
lr.fit(x_train,y_train)
y_lr = lr.predict(x_test)
score.append(lr.score(x_test,y_test))
print('                 Logistic Regression      ')
print(cross_val_score(lr,x_train,y_train,cv=3))
print(f1(y_test,y_lr,average='micro'))

#LDA
lda = LinearDiscriminantAnalysis(solver='svd')
lda.fit(x_train,y_train)
y_lda = lda.predict(x_test)
score.append(lda.score(x_test,y_test))
print('                 Linear Discriminant Analysis         ')
print(cross_val_score(lda,x_train,y_train,cv=3))
print(f1(y_test,y_lda,average='micro'))

#Perceptron
per = Perceptron(penalty='l1',alpha=0.001)
per.fit(x_train,y_train)
y_per = per.predict(x_test)
score.append(per.score(x_test,y_test))
print('                 Perceptron Classifier         ')
print(cross_val_score(per,x_train,y_train,cv=3))
print(f1(y_test,y_per,average='micro'))

#SVM
svm = SVC(C=100,kernel='poly',degree=4,coef0 = 0)
svm.fit(x_train,y_train)
y_svm = svm.predict(x_test)
score.append(svm.score(x_test,y_test))
print('                Support Vector Machine           ')
print(f1(y_test,y_svm,average='micro'))
print(cross_val_score(svm,x_train,y_train,cv=3))

#Naives Bayes Algorithm
nb = MultinomialNB(alpha=0.4)
nb.fit(x_train,y_train)
y_nb = nb.predict(x_test)
score.append(nb.score(x_test,y_test))
print('                Naives Bayes Algorithm            ')
print(cross_val_score(nb,x_train,y_train,cv=3))
print(f1(y_test,y_nb,average='micro'))

##Comparison of above classification algorithms
#f1 = plt.figure()
#n = np.arange(7)
#plt.bar(n,score)
#plt.xticks(n,('Decision Tree','kNN','Logistic','LDA','Perceptron','SVM','Naives Bayes'))
#plt.xlabel('Classification Algorithm')
#plt.ylabel('F1 score')
#plt.title('Comparison of Classification Algorithms')
#plt.show()


#Analysis of the chosen model
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    f2 = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

#cnf_matrix = confusion_matrix(y_test,y_svm)
#plot_confusion_matrix(cnf_matrix,[1,2,3,4,5,6,7],normalize=False)

svm2 = SVC(C=100,kernel='poly',degree=5,coef0 = 7,gamma=0.38,class_weight={1:0.05,2:0.05,3:2,4:0.05,5:0.05,6:0.05,7:1} )
svm2.fit(pca_trainx,pca_trainy)
y_svm2 = svm2.predict(pca_testx)
print('                Support Vector Machine           ')
mat = confusion_matrix(pca_testy,y_svm2)
plot_confusion_matrix(mat,[1,2,3,4,5,6,7],normalize=False)

ada = AdaBoostClassifier((SVC(C=350,kernel='poly',degree=2,coef0=0,class_weight={1:1,2:1,3:10,4:1,5:1,6:1,7:1})),algorithm='SAMME',learning_rate=1)
ada.fit(x_train,y_train)
y_ada = ada.predict(x_test)
print(ada.score(x_test,y_test))
mat2 = confusion_matrix(y_test,y_ada)
plot_confusion_matrix(mat2,[1,2,3,4,5,6,7])