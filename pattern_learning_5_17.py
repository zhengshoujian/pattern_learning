#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:55:49 2019

@author: zhengshoujian
"""

import model
import tensorflow as tf
import  os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from cifardata import CifarData

CIFAR_DIR = "./cifar-10-batches-py"
model_save_path = './model/'

def load_data( filename ):
    '''read data from data file'''
    with open( filename, 'rb' ) as f:
        data = pickle.load( f, encoding='bytes' ) # python3 需要添加上encoding='bytes'
        return data[b'data'], data[b'labels'] # 并且 在 key 前需要加上 b



train_filename_normal = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 5)]
train_filename_pattern = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(5, 6)]
test_filename = [os.path.join(CIFAR_DIR, 'test_batch')]

train_data_normal = CifarData( train_filename_normal, True )
train_data_pattern = CifarData( train_filename_pattern, True )
test_data = CifarData( test_filename, False )


batch_size = 4
x = tf.placeholder( tf.float32, [None, 3072] )
y = tf.placeholder( tf.int64, [None] ) 

x_image = tf.reshape( x, [-1, 3, 32, 32] )
x_image = tf.transpose( x_image, perm= [0, 2, 3, 1] )

y_ = model.my_net(x_image,use_bn = True if batch_size>1 else False)


# 使用交叉熵 设置损失函数
loss = tf.losses.sparse_softmax_cross_entropy( labels = y, logits = y_ )
# 该api,做了三件事儿 1. y_ -> softmax 2. y -> one_hot 3. loss = ylogy

# 预测值 获得的是 每一行上 最大值的 索引.注意:tf.argmax()的用法,其实和 np.argmax() 一样的
predict = tf.argmax( y_, 1 )
# 将布尔值转化为int类型,也就是 0 或者 1, 然后再和真实值进行比较. tf.equal() 返回值是布尔类型
correct_prediction = tf.equal( predict, y )
# 比如说第一行最大值索引是6,说明是第六个分类.而y正好也是6,说明预测正确



accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float64) )

with tf.name_scope( 'train_op_small' ): # tf.name_scope() 这个方法的作用不太明白(有点迷糊!)
    train_op_small = tf.train.AdamOptimizer(1e-4).minimize( loss ) # 将 损失函数 降到 最低
    
with tf.name_scope( 'train_op_big' ): # tf.name_scope() 这个方法的作用不太明白(有点迷糊!)
    train_op_big = tf.train.AdamOptimizer(1e-3).minimize( loss ) # 将 损失函数 降到 最低

# 初始化变量
init = tf.global_variables_initializer()

epoch = 10
train_steps_normal = int(epoch*20000/batch_size)
test_steps = 300

train_step_pattern = 10000

time_now = time.strftime('%Y.%m.%d',time.localtime(time.time()))
loss_sum = []
loss_before = 5.0
loss_sum.append(loss_before)
count_iter = 0
count_no_iter = 0
count_1 = 0
count_2 = 0

batch_count_max = 7
batch_count = 0

length_of_loss_aver = 1

test_acc_val = []

list_count_1 =[]
loss_val_1 = []

list_count_2 =[]
loss_val_2 =[]

pattern1 = "loss_before>loss_val"
pattern2 = "loss_before<loss_val"

#f = open("output",'a+')
update_1 = True
updata_2 = True
f = open('result.txt','a+')
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run( init ) # 注意: 这一步必须要有!!
	# 开始训练
	for i in range( train_steps_normal ):
		# 得到batch
		batch_data, batch_labels = train_data_normal.next_batch( batch_size )
		
		loss_val, acc_val = sess.run( [loss, accuracy], feed_dict={x:batch_data, y:batch_labels} )
		loss_before = loss_val
		sess.run([train_op_big], feed_dict={x:batch_data, y:batch_labels} )
		
		if(count_1%200==0):
			list_count_1.append(count_1)
			loss_val_1.append(loss_val)
		count_1+=1
		i+=1
		
		while(batch_count<batch_count_max):
			loss_val, acc_val = sess.run( [loss, accuracy], feed_dict={x:batch_data, y:batch_labels} )
			if(loss_before>loss_val):
				batch_count = batch_count_max
			else:
				#batch_count not too much
				batch_count+=1
				sess.run([train_op_big], feed_dict={x:batch_data, y:batch_labels} )
				
				if(count_1%20==0):
					list_count_1.append(count_1)
					loss_val_1.append(loss_val)
				count_1+=1
				i+=1
			
				loss_before = loss_val


		# 每 500 次 输出一条信息
		if ( i+1 ) % 1 == 0:
			print('\n[Train] Step: %d, loss: %4.8f, acc: %4.8f' % ( i+1, loss_val, acc_val ))
		# 每 5000 次 进行一次 测试
	test_acc=0;
	test_data = CifarData( test_filename, False )
	all_test_acc_val = []
	for j in range( test_steps ):
		test_batch_data, test_batch_labels = test_data.next_batch( batch_size )
		test_acc_val = sess.run( [accuracy], feed_dict={ x:test_batch_data, y:test_batch_labels } )
		all_test_acc_val.append( test_acc_val )
	test_acc = np.mean( all_test_acc_val )
	local_time = time.asctime(time.localtime(time.time()))
	line = "pattern learning: "+pattern1 +' '+ str(time.time()) + ' batch_size = '+str(batch_size)+' batch_count_max = '+str(batch_count_max) + ' pattern_learning_test acc ' + str(test_acc) +" batch_size = "+str(batch_size) +'\n'
	print(line)
	f.write(line)
	print('----------------------------------------------------')
	print('\n[Test ] Step: %d, acc: %4.5f' % ( (i+1), test_acc ))
	print('----------------------------------------------------')


	#save_path = saver.save(sess, model_save_path)
	#print("Model saved in file: %s" % save_path)
	#line = time_now + ' ' + str(count_iter) + ' ' + str(count_no_iter) + ' ' + str(test_acc) + ' ' + str(loss_no_cout[-1]) + '\n'
	#f.write(line)


	save_path = saver.save(sess, model_save_path)
	print("Model saved in file: %s" % save_path)
	#line = time_now + ' ' + str(count_iter) + ' ' + str(count_no_iter) + ' ' + str(test_acc) + ' ' + '\n'
	#f.write(line)
	#f.close()


with tf.Session() as sess:
    sess.run( init ) # 注意: 这一步必须要有!!
    # 开始训练
    for i in range( train_steps_normal):
        # 得到batch
        batch_data, batch_labels = train_data_normal.next_batch( batch_size )
        # 获得 损失值, 准确率

        loss_val, acc_val = sess.run( [loss, accuracy], feed_dict={x:batch_data, y:batch_labels} )

        sess.run([train_op_big], feed_dict={x:batch_data, y:batch_labels} )
            
        count_2+=1
        
        #data for figure
        if(count_2%200==0):
            list_count_2.append(count_2)
            loss_val_2.append(loss_val)

        # 每 500 次 输出一条信息
        if ( i+1 ) % 1 == 0:
            print('\n[Train] Step: %d, loss: %4.8f, acc: %4.8f' % ( i+1, loss_val, acc_val ))
        # 每 5000 次 进行一次 测试
    
    test_acc=0
    test_data = CifarData( test_filename, False )
    all_test_acc_val = []
    for j in range( test_steps ):
        test_batch_data, test_batch_labels = test_data.next_batch( batch_size )
        test_acc_val = sess.run( [accuracy], feed_dict={ x:test_batch_data, y:test_batch_labels } )
        all_test_acc_val.append( test_acc_val )
    test_acc = np.mean( all_test_acc_val )
    local_time = time.asctime(time.localtime(time.time()))
    line ="normal learning:"+' '*23 + str(time.time()) +' batch_size = '+str(batch_size)+" "*20 +' normal_learning_test acc = ' + str(test_acc)+" batch_size = "+str(batch_size) + '\n'
    print(line)
    f.write(line)
            
            
    save_path = saver.save(sess, model_save_path)
    print("Model saved in file: %s" % save_path)
    #line = time_now + ' ' + str(count_iter) + ' ' + str(count_no_iter) + ' ' + str(test_acc) + ' ' + '\n'
    #f.write(line)
    #f.close()
    f.close()


    
plt.figure()
plt.plot(list_count_1,loss_val_1,color = 'b',linestyle = '-',label='pattern_learning')
plt.plot(list_count_2,loss_val_2,color = 'y',linestyle = '--',label='normal_learning')
plt.savefig("/home/dongdong/zweistein/cifar10/image_output/"+str(time.time())+str(batch_count_max)+".png")

