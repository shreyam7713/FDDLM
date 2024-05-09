import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

def graph_drawing(model_history, title, model_name):
    fig, axs = plt.subplots(len(model_history), figsize=(8,12))
    for i in range(len(model_history)):
        ax = axs[i]
        history = model_history[i]
        x = np.arange(len(history.history["val_acc"]))
        ax.plot(x, history.history["loss"], label="loss")
        ax.plot(x, history.history["val_loss"], label="val_loss")

        ax.plot(x, history.history["acc"], label="accuracy")
        ax.plot(x, history.history["val_acc"], label="val_accuracy")
        subtitle = title + f"{i+1}"
        ax.set_title(subtitle)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"./graphs/{model_name} graph")

def heatmap_drawing(confusion_matrices, title, model_name):
    fig, axs = plt.subplots(len(confusion_matrices), figsize=(10,20))
    for i,cm in enumerate(confusion_matrices):
        ax = axs[i]
        sns.heatmap(cm, annot=True, ax=ax, fmt='d')
        ax.set_title(f"{title} {i+1}")
    plt.tight_layout()
    plt.savefig(f"./heatmaps/{model_name} heatmap")

def get_model_performance(X, y, model_func, epochs=100):
    full_history =  []
    confusion_matrices= []
    for i in range(len(X)):
        X_test, y_test = X[i], y[i]
        X_train, y_train = [ X[j] for j in range(len(X)) if j != i], [y[j] for j in range(len(X)) if j != i]
        X_train, y_train = np.vstack(X_train), np.hstack(y_train)
        one_hot_train_labels = to_categorical(y_train)
        one_hot_test_labels = to_categorical(y_test)
        model = model_func(X_train)
        history = model.fit(X_train, one_hot_train_labels, epochs= epochs, batch_size=64, validation_data = (X_test, one_hot_test_labels))
        full_history.append(history)
        pred = model.predict(X_test)
        pred = np.argmax(pred, axis=1)
        confusion_matrix = metrics.confusion_matrix(pred, y_test)
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
    model.add( layers.Flatten())
    model.add( layers.Dense(100, activation='relu', name="Dense_2"))
    model.add( layers.Dense(5, activation='softmax', name='output'))
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

try:
    physical_device = tf.config.experimental.list_physical_devices('GPU')
    print(f'Device found : {physical_device}')
except Exception as e:
    print(f"An error occurred: {e}")
    print("Continuing program execution...")

print("FDDLM data loaded\n")
X1 = np.load("../COPD_dataset/hold-out-validation/cp_raw_data.npy")
X2 = np.load("../COPD_dataset/hold-out-validation/md1_raw_data.npy")
X3 = np.load("../COPD_dataset/hold-out-validation/md2_raw_data.npy")
X4 = np.load("../COPD_dataset/hold-out-validation/vb_raw_data.npy")

y = [X1[:,-1],X2[:,-1],X3[:,-1],X4[:,-1],]

X = [X1[:,:-1], X2[:,:-1], X3[:,:-1], X4[:,:-1]]

del X1, X2, X3, X4
print("FDDLM X,y created\n")

fddlm_history, fddlm_confusion_matrices = get_model_performance(model_func= FDDLM_Model, X= X, y= y, epochs=100)
np.save("./heatmaps/fddlm_cfs.npy" , fddlm_confusion_matrices)
save_history("./history/fddlm_history.npy", fddlm_history)

fddlm_history, fddlm_confusion_matrices = get_model_performance(model_func= FDDLM_Model, X= X, y= y, epochs=500)
np.save("./heatmaps/fddlm_cfs_500.npy" , fddlm_confusion_matrices)
save_history("./history/fddlm_history_500.npy", fddlm_history)
