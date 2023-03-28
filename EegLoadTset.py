# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:49:39 2023

@author: Administrator
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import os

#model.save('E:/代码/脑电深度学习模型/cnn_maxpool_model_1/my_model')

new_model = tf.keras.models.load_model('E:/代码/脑电深度学习模型/cnn_maxpool_model_1/my_model')
new_model.summary()

rhythm_all_sub_all = loadmat("E:/数据集/psd_moving_all.mat")
label_2class = loadmat("E:/数据集/rhythm_all/label.mat")

features = rhythm_all_sub_all['data_all']
labels = label_2class['label_all']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

# Evaluate the restored model
loss, acc = new_model.evaluate(X_test, y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(X_test).shape)