import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping

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

def get_model_performance(X, y, model_func, epochs=100):
    full_history =  []
    confusion_matrices= []
    for i in range(len(X)):
        X_test, y_test = X[i], y[i]
        X_train, y_train = [ X[j] for j in range(len(X)) if j != i], [y[j] for j in range(len(X)) if j != i]
        X_train, y_train = np.vstack(X_train), np.hstack(y_train)
        
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        Y_train = y_train.reshape(y_train.shape[0],1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_test = y_test.reshape(y_test.shape[0], 1)
                
        model = model_func(X_train)
        earlystopping = EarlyStopping(patience=50, monitor="val_accuracy", restore_best_weights=True, mode='max')
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
        average['accuracy'] += history[i].history['accuracy'][last_epoch]
        average['loss'] += history[i].history['loss'][last_epoch]
        average['val_accuracy'] += history[i].history['val_accuracy'][last_epoch]
        average['val_loss'] += history[i].history['val_loss'][last_epoch]
    
        best['accuracy'] = max(best['accuracy'],history[i].history['accuracy'][last_epoch])
        best['loss'] = min(best['loss'] , history[i].history['loss'][last_epoch])
        best['val_accuracy'] = max(best['val_accuracy'], history[i].history['val_accuracy'][last_epoch])
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

def CNN_Model(data):
    model = keras.Sequential(name="CNN_Sequential")
    model.add(layers.Input(shape = data.shape[1:]))
    model.add(layers.Conv1D(128, 3, activation='relu', name='conv_1'))
    model.add( layers.Conv1D(64, 4, activation="softmax", name="conv_2") )
    model.add(layers.Flatten())
    model.add( layers.Dense(64, activation="relu", name="dense_1") )
    model.add( layers.Dense(128, activation="relu", name="dense_2") )
    model.add( layers.Dense(6, activation="softmax", name="output") )
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    return model

def LSTM_Model(data):
    model = keras.Sequential(name="LSTM_Sequential")
    print(data.shape[1:])
    model.add( layers.Input(shape=data.shape[1:]) )
    model.add(layers.LSTM(32, activation='relu', name='LSTM'))
    model.add( layers.Dense(64, activation="relu", name="dense_1") )
    model.add( layers.Dense(128, activation="relu", name="dense_2") )
    model.add( layers.Dense(6, activation="softmax", name="output") )
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    return model

def CNN_LSTM_Model(data):
    model = keras.Sequential(name="CNN_Sequential")
    print(data.shape[1:])
    model.add( layers.Input(shape=data.shape[1:]) )
    model.add( layers.Conv1D(64, 3, activation="relu", name="conv_1") )
    model.add( layers.Conv1D(32, 4, activation="relu", name="conv_2") )
    
    model.add(layers.LSTM(20, activation="tanh", name="LSTM"))

    model.add( layers.Dense(64, activation="relu", name="dense_1") )
    model.add( layers.Dense(128, activation="relu", name="dense_2") )
    model.add( layers.Dense(6, activation="softmax", name="output") )
    
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    return model

def FDDLM_Model(data):
    model = keras.Sequential(name="FDDLM_Sequential")
    print(data.shape[1:])
    model.add( layers.Input(shape=data.shape[1:]) )
    model.add( layers.Dense(300, activation='relu', name="Dense_1"))
    model.add(layers.Flatten())
    model.add( layers.Dense(100, activation='relu', name="Dense_2"))
    model.add( layers.Dense(6, activation='softmax', name='output'))
    
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    return model

def save_history(filename, history):
    np_history = []
    for i in range(len(history)):
        his = history[i]
        np_history.append(his.history)
    np.save(filename, np_history)

X1 = np.load("../COPD_dataset/hold-out-validation/cp_raw_data.npy")
X2 = np.load("../COPD_dataset/hold-out-validation/md1_raw_data.npy")
X3 = np.load("../COPD_dataset/hold-out-validation/md2_raw_data.npy")
X4 = np.load("../COPD_dataset/hold-out-validation/vb_raw_data.npy")

y = [X1[:,-1],X2[:,-1],X3[:,-1],X4[:,-1],]

X = [X1[:,:-1], X2[:,:-1], X3[:,:-1], X4[:,:-1]]

cnn_history, cnn_confusion_matrices = get_model_performance(model_func= CNN_Model, X= X, y= y, epochs=100)
get_history_summary(cnn_history,"CNN K-fold","CNN")
graph_drawing(cnn_history,"CNN K-fold","CNN")
heatmap_drawing(cnn_confusion_matrices,"CNN K-fold","CNN")

lstm_history, lstm_confusion_matrices = get_model_performance(model_func= LSTM_Model, X= X, y= y, epochs=100)
get_history_summary(lstm_history,"LSTM K-fold","LSTM")
graph_drawing(lstm_history,"LSTM K-fold","LSTM")
heatmap_drawing(lstm_confusion_matrices, "LSTM K-fold","LSTM")

cnn_lstm_history, cnn_lstm_confusion_matrices = get_model_performance(model_func= CNN_LSTM_Model, X= X, y= y, epochs=100)
get_history_summary(cnn_lstm_history,"CNN LSTM K-fold")
graph_drawing(cnn_lstm_history,"CNN LSTM K-fold")
heatmap_drawing(cnn_lstm_confusion_matrices,"CNN LSTM K-fold")

fddlm_history, fddlm_confusion_matrices = get_model_performance(model_func= FDDLM_Model, X= X, y= y, epochs=100)
get_history_summary(fddlm_history,"FDDLM K-fold")
graph_drawing(fddlm_history,"FDDLM K-fold")
heatmap_drawing(fddlm_confusion_matrices,"FDDLM K-fold")

save_history("./history/cnn_history.npy", cnn_history)
save_history("./history/lstm_history.npy", lstm_history)
save_history("./history/cnn_lstm_history.npy", cnn_lstm_history)
save_history("./history/fddlm_history.npy", fddlm_history)