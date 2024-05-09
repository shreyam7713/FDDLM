import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

def test_model(X , y, model):
    cfs = []
    for i in range(len(X)):
        X_test, y_test = X[i], y[i]
        X_train, y_train = [ X[j] for j in range(len(X)) if j != i], [y[j] for j in range(len(X)) if j != i]
        X_train, y_train = np.vstack(X_train), np.hstack(y_train)
    
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        orignal = y_test
        cf = metrics.confusion_matrix(pred,orignal)
        cfs.append(cf)
    return cfs

def heatmap_drawing(confusion_matrices, title,model_name):
    n = len(confusion_matrices)
    fig, axs = plt.subplots(n, figsize=(8,12))
    for i,cm in enumerate(confusion_matrices):
        ax = axs[i]
        sns.heatmap(cm, annot=True, ax=ax,fmt='d')
        ax.set_title(f"{title} {i+1}")
    plt.tight_layout()
    plt.savefig(f"./heatmaps/{model_name} heatmap")

def save_best_results(params, score, model_name, filename):
    with open(filename,'a') as file:
        file.write(f"{model_name}\n")
        file.write(f"Best score in {model_name}: {score}\n")
        file.write(f"Params used: {params}\n")
        file.write("\n\n")

X1 = np.load("../COPD_dataset/hold-out-validation/cp_raw_data.npy")
X2 = np.load("../COPD_dataset/hold-out-validation/md1_raw_data.npy")
X3 = np.load("../COPD_dataset/hold-out-validation/md2_raw_data.npy")
X4 = np.load("../COPD_dataset/hold-out-validation/vb_raw_data.npy")

y = [X1[:,-1],X2[:,-1],X3[:,-1],X4[:,-1],]

X = [X1[:,:-1], X2[:,:-1], X3[:,:-1], X4[:,:-1]]

np_X = np.vstack(X)
np_y = np.hstack(y)

knn = RandomizedSearchCV( KNeighborsClassifier(algorithm='auto'), {
    'n_neighbors' : range(1,10,1),
    'weights': ['uniform', 'distance'],
}, cv = 5, return_train_score= False, n_iter= 10)
knn.fit(np_X, np_y)
save_best_results(score = knn.best_score_, params = knn.best_params_, filename = "ML_best_performances.txt", model_name = "KNN")

svm = RandomizedSearchCV( SVC(gamma='auto'),{
    'C': range(1,100),
    'kernel': ['rbf','linear'],
}, cv= 5, return_train_score= False, n_iter=20)
svm.fit(np_X, np_y)
save_best_results(score = svm.best_score_, params = svm.best_params_, file_name = "ML_best_performances.txt", model_name = "SVM")

rfc = RandomizedSearchCV( RandomForestClassifier(),{
    'n_estimators':range(1,200),
    'criterion': ["gini", "entropy", "log_loss"],
    'max_depth':range(100,600,10)
}, cv=5, return_train_score=False, n_iter=50)
rfc.fit(np_X,np_y)
save_best_results(score = rfc.best_score_, params = rfc.best_params_, file_name = "ML_best_performances.txt", model_name = "RFC")

params = knn.best_params_
knn = KNeighborsClassifier(algorithm='auto', weights = params['weights'], n_neighbors = params['n_neighbors'])

params = svm.best_params_
svm = SVC(gamma='auto', C = params['C'], kernel=params['kernel'])

params = rfc.best_params_
rfc = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], criterion=params['criterion'])

knn_cfs = test_model(X, y, knn)

svm_cfs = test_model(X, y, svm)

rfc_cfs = test_model(X, y, rfc)

heatmap_drawing(knn_cfs, "KNN kfold","KNN")

heatmap_drawing(svm_cfs, "SVM k-fold","SVM")

heatmap_drawing(rfc_cfs, "RFC kfold","RFC")