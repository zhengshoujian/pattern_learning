#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:55:49 2019

@author: zhengshoujian
"""


import tensorflow as tf


def focal_loss(predictions,labels,alpha,gamma):
	zeros = tf.zeros_like(predictions,dtype = predictions.dtype)
	pos_corr = tf.where(labels>zeros,labels - predictions,zeros)
	neg_corr = tf.where(labels>zeros,zeros,predictions)
	fl_loss = -alpha*(pos_corr**alpha)*tf.log(predictions) - (1 - alpha)*(neg_corr**gamma)*tf.log(1.0 - predictions)
	return tf.reduce_sum(fl_loss)

