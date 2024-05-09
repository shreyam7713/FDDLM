import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

X = np.load("../COPD_dataset/k-fold-validation/X2_data-002.npy")
y = X[:, -1]
y = y.astype(np.int64)
X = X[:,:-1]
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

def test_model(X , y, model, k_fold=5):
    kfold = KFold(n_splits=k_fold, shuffle=False)
    cfs = []
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
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


knn = RandomizedSearchCV( KNeighborsClassifier(algorithm='auto'), {
    'n_neighbors' : range(1,10,1),
    'weights': ['uniform', 'distance'],
}, cv = 5, return_train_score= False, n_iter= 10)
knn.fit(X, y)
save_best_results(score = knn.best_score_, params = knn.best_params_, filename = "ML_best_performances.txt", model_name = "KNN")

svm = RandomizedSearchCV( SVC(gamma='auto'),{
    'C': range(1,100),
    'kernel': ['rbf','linear'],
}, cv= 5, return_train_score= False, n_iter=20)
svm.fit(X, y)
save_best_results(score = svm.best_score_, params = svm.best_params_, file_name = "ML_best_performances.txt", model_name = "SVM")

rfc = RandomizedSearchCV( RandomForestClassifier(),{
    'n_estimators':range(1,200),
    'criterion': ["gini", "entropy", "log_loss"],
    'max_depth':range(100,600,10)
}, cv=5, return_train_score=False, n_iter=50)
rfc.fit(X,y)
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