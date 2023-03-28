# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 22:09:16 2023

@author: Administrator
"""

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('E:/代码/脑电深度学习模型/cnn_maxpool_model_1/my_model')
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)