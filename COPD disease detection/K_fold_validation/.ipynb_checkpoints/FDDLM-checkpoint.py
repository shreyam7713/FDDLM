import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

def graph_drawing(model_history, title, model_name):
    fig, axs = plt.subplots(len(model_history), figsize=(8,12))
    for i in range(len(model_history)):
        ax = axs[i]
        history = model_history[i]
        x = np.arange(len(history.history["val_accuracy"]))
        ax.plot(x, history.history["loss"], label="loss")
        ax.plot(x, history.history["val_loss"], label="val_loss")

        ax.plot(x, history.history["accuracy"], label="accuracy")
        ax.plot(x, history.history["val_accuracy"], label="val_accuracy")
        subtitle = title + f"{i+1}"
        ax.set_title(subtitle)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"./graphs/{model_name} graph")

def heatmap_drawing(confusion_matrices, title, model_name):
    fig, axs = plt.subplots(len(confusion_matrices), figsize=(8,12))
    for i,cm in enumerate(confusion_matrices):
        ax = axs[i]
        sns.heatmap(cm, annot=True, ax=ax, fmt='d')
        ax.set_title(f"{title} {i+1}")
    plt.tight_layout()
    plt.savefig(f"./heatmaps/{model_name} heatmap")

def get_model_performance(X, y, model_func, k_splits = 5,epochs=100):
    kfold = KFold(n_splits= k_splits, shuffle=False)
    full_history =  []
    confusion_matrices= []
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = model_func(X_train)
        earlystopping = EarlyStopping(patience=30, monitor="val_acc", restore_best_weights=True, mode='max')
        history = model.fit(X_train, y_train, epochs= epochs, callbacks = [earlystopping], validation_data = (X_test, y_test))
        full_history.append(history)
        pred = model.predict(X_test)
        orignal = y_test
        pred = np.argmax(pred, axis=1)
        confusion_matrix = metrics.confusion_matrix(pred, orignal)
        confusion_matrices.append(confusion_matrix)

    return full_history, confusion_matrices

def get_history_summary(history ,title ,model_name):
    n = len(history)
    average = {'accuracy': 0, 'loss':0,'val_accuracy':0, 'val_loss':0 }
    best = {'accuracy': 0, 'loss':10000000,'val_accuracy':0, 'val_loss':100000000 }
    for i in range(n):
        last_epoch = len(history[i].history['loss']) - 1
        average['accuracy'] += history[i].history['acc'][last_epoch]
        average['loss'] += history[i].history['loss'][last_epoch]
        average['val_accuracy'] += history[i].history['val_acc'][last_epoch]
        average['val_loss'] += history[i].history['val_loss'][last_epoch]
    
        best['accuracy'] = max(best['accuracy'],history[i].history['acc'][last_epoch])
        best['loss'] = min(best['loss'] , history[i].history['loss'][last_epoch])
        best['val_accuracy'] = max(best['val_accuracy'], history[i].history['val_acc'][last_epoch])
        best['val_loss'] = min(best['val_loss'], history[i].history['val_loss'][last_epoch])
        
    average['accuracy'] = average['accuracy']/n
    average['loss'] = average['loss']/n
    average['val_accuracy'] = average['val_accuracy']/n
    average['val_loss'] = average['val_loss']/n

    labels = list(average.keys())
    average_values = list(average.values())
    best_values = list(best.values())

    bar_width = 0.35
    index = np.arange(len(labels))
    bar1 = plt.bar(index,average_values, bar_width, color= '#ee8b0d', label= 'Average')
    bar2 = plt.bar(index + bar_width, best_values, bar_width, color= 'b', label= 'Best')
    for i, v in enumerate(average_values):
        plt.text(i, v + 0.01, str(round(v, 2)), ha='center', va='bottom') 
    for i, v in enumerate(best_values):
        plt.text(i + bar_width, v + 0.01, str(round(v, 2)), ha='center', va='bottom') 

    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title(f"Comparison of Average and Best Metrics for {title}")
    plt.xticks([i + bar_width / 2 for i in index])
    plt.xticks([i + bar_width / 2 for i in index], labels)
    plt.legend()
    plt.savefig(f"./summary/{model_name} summary")

def save_history(filename, history):
    np_history = []
    for i in range(len(history)):
        his = history[i]
        np_history.append(his.history)
    np.save(filename, np_history)

def FDDLM_Model(data):
    model = keras.Sequential(name="FDDLM_Sequential")
    print(data.shape[1:])
    model.add( layers.Input(shape=data.shape[1:]) )
    model.add( layers.Dense(300, activation='relu', name="Dense_1"))
    model.add(layers.Flatten())
    model.add( layers.Dense(100, activation='relu', name="Dense_2"))
    model.add( layers.Dense(5, activation='softmax', name='output'))
    
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    return model

try:
    physical_device = tf.config.experimental.list_physical_devices('GPU')
    print(f'Device found : {physical_device}')
except Exception as e:
    print(f"An error occurred: {e}")
    print("Continuing program execution...")

X = np.load("../COPD_dataset/k-fold-validation/X2_data-002.npy")
y = X[:, -1]
y = y.astype(np.int64)
X = X[:,:-1]
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

X = X.reshape(X.shape[0], X.shape[1], 1)
y = y.reshape(y.shape[0], 1)
print("FDDLM X,y created\n")

fddlm_history, fddlm_confusion_matrices = get_model_performance(model_func= FDDLM_Model, X= X, y= y, epochs=100)
save_history("./history/fddlm_history.npy", fddlm_history)
np.save("./heatmaps/fddlm_cfs.npy" , fddlm_confusion_matrices)