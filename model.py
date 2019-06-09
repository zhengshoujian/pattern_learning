import tensorflow as tf

def batch_norm_layer(value,is_training=False,name='batch_norm'):
    '''
    批量归一化  返回批量归一化的结果
    
    args:
        value:代表输入，第一个维度为batch_size
        is_training:当它为True，代表是训练过程，这时会不断更新样本集的均值与方差。当测试时，要设置成False，这样就会使用训练样本集的均值和方差。
              默认测试模式
        name：名称。
    '''
    if is_training is True:
        #训练模式 使用指数加权函数不断更新均值和方差
        return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = True)
    else:
        #测试模式 不更新均值和方差，直接使用
        return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = False)

def model(x_image):# pool = 4 lay = 12
    conv1_1 = tf.layers.conv2d( x_image,32,  ( 3, 3 ),padding = 'same', activation = tf.nn.relu,name = 'conv1_1')
    conv1_2 = tf.layers.conv2d( conv1_1,32,( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv1_2')
    pooling1 = tf.layers.max_pooling2d( conv1_2, ( 2, 2 ),( 2, 2 ),name='pool1')

    conv2_1 = tf.layers.conv2d( pooling1,64,( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv2_1')
    conv2_2 = tf.layers.conv2d( conv2_1,64,( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv2_2')
    pooling2 = tf.layers.max_pooling2d( conv2_2,( 2, 2 ),( 2, 2 ), name='pool2')

    conv3_1 = tf.layers.conv2d( pooling2,128,  ( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv3_1')
    conv3_2 = tf.layers.conv2d( conv3_1,128, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv3_2')
    pooling3 = tf.layers.max_pooling2d( conv3_2,( 2, 2 ),( 2, 2 ),name='pool3')
        
    conv4_1 = tf.layers.conv2d( pooling3,256,  ( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv4_1')
    conv4_2 = tf.layers.conv2d( conv4_1,256, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv4_2')
    pooling4 = tf.layers.max_pooling2d( conv4_2,( 2, 2 ),( 2, 2 ),name='pool4')
    
    conv5_1 = tf.layers.conv2d( pooling4,512,  ( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv5_1')
    conv5_2 = tf.layers.conv2d( conv5_1,512, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv5_2')
    pooling5 = tf.layers.max_pooling2d( conv5_2,( 2, 2 ),( 2, 2 ),name='pool5')
    # 展平
    flatten  = tf.contrib.layers.flatten(pooling5)
    y_ = tf.layers.dense(flatten, 10)
    return y_


def model2(x_image):# pool = 4 lay = 17
    conv1_1 = tf.layers.conv2d( x_image,32, ( 3, 3 ),padding = 'same', activation = tf.nn.relu,name = 'conv1_1')
    conv1_2 = tf.layers.conv2d( conv1_1,32,( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv1_2')
    conv1_3 = tf.layers.conv2d( conv1_2,32,( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv1_3')
    pooling1 = tf.layers.max_pooling2d( conv1_3, ( 2, 2 ),( 2, 2 ),name='pool1')

    conv2_1 = tf.layers.conv2d( pooling1,64,( 3, 3), padding = 'same',activation = tf.nn.relu,name = 'conv2_1')
    conv2_2 = tf.layers.conv2d( conv2_1,64,( 3, 3 ),padding  = 'same',activation = tf.nn.relu,name = 'conv2_2')
    conv2_3 = tf.layers.conv2d( conv2_2,32,( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv2_3')
    pooling2 = tf.layers.max_pooling2d( conv2_3,( 2, 2 ),( 2, 2 ), name='pool2')

    conv3_1 = tf.layers.conv2d( pooling2,128,( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv3_1')
    conv3_2 = tf.layers.conv2d( conv3_1,128, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv3_2')
    conv3_3 = tf.layers.conv2d( conv3_2,32,( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv3_3')
    pooling3 = tf.layers.max_pooling2d( conv3_3,( 2, 2 ),( 2, 2 ),name='pool3')
        
    conv4_1 = tf.layers.conv2d( pooling3,256,( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv4_1')
    conv4_2 = tf.layers.conv2d( conv4_1,256, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv4_2')
    conv4_3 = tf.layers.conv2d( conv4_2,256, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv5_2')
    pooling4 = tf.layers.max_pooling2d( conv4_3,( 2, 2 ),( 2, 2 ),name='pool4')
    
    conv5_1 = tf.layers.conv2d( pooling4,512,( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv5_1')
    conv5_2 = tf.layers.conv2d( conv5_1,512, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv5_2')
    conv5_3 = tf.layers.conv2d( conv5_2,512, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv5_3')
    pooling5 = tf.layers.max_pooling2d( conv5_3,( 2, 2 ),( 2, 2 ),name='pool5')
    # 展平
    flatten  = tf.contrib.layers.flatten(pooling5)
    y_ = tf.layers.dense(flatten, 10)
    return y_

def vgg(x_image):# pool = 4 lay = 12
    conv1_1 = tf.layers.conv2d( x_image,64,  ( 3, 3 ),padding = 'same', activation = tf.nn.relu,name = 'conv1_1')
    conv1_2 = tf.layers.conv2d( conv1_1,64,( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv1_2')
    pooling1 = tf.layers.max_pooling2d( conv1_2, ( 2, 2 ),( 2, 2 ),name='pool1')

    conv2_1 = tf.layers.conv2d( pooling1,128,( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv2_1')
    conv2_2 = tf.layers.conv2d( conv2_1,128,( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv2_2')
    pooling2 = tf.layers.max_pooling2d( conv2_2,( 2, 2 ),( 2, 2 ), name='pool2')

    conv3_1 = tf.layers.conv2d( pooling2,256,  ( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv3_1')
    conv3_2 = tf.layers.conv2d( conv3_1,256, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv3_2')
    #conv3_3 = tf.layers.conv2d( conv3_2,256, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv3_3')
    pooling3 = tf.layers.max_pooling2d( conv3_2,( 2, 2 ),( 2, 2 ),name='pool3')
        
    conv4_1 = tf.layers.conv2d( pooling3,512,  ( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv4_1')
    conv4_2 = tf.layers.conv2d( conv4_1,512, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv4_2')
    #conv4_3 = tf.layers.conv2d( conv4_2,512, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv4_3')
    pooling4 = tf.layers.max_pooling2d( conv4_2,( 2, 2 ),( 2, 2 ),name='pool4')
    
    conv5_1 = tf.layers.conv2d( pooling4,512,  ( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv5_1')
    conv5_2 = tf.layers.conv2d( conv5_1,512, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv5_2')
    #conv5_3 = tf.layers.conv2d( conv5_2,512, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv5_3')
    pooling5 = tf.layers.max_pooling2d( conv5_2,( 2, 2 ),( 2, 2 ),name='pool5')
    # 展平
    flatten  = tf.contrib.layers.flatten(pooling5)
    y_ = tf.layers.dense(flatten, 10)

    return y_

def my_net(x_image,use_bn):# pool = 4 lay = 12
    conv1_1 = tf.layers.conv2d( x_image,64,  ( 3, 3 ),padding = 'same', activation = tf.nn.relu,name = 'conv1_1')
    bn1 = batch_norm_layer(conv1_1,is_training=use_bn,name='batch_norm1_1')
    #conv1_2 = tf.layers.conv2d( conv1_1,64,( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv1_2')
    pooling1 = tf.layers.max_pooling2d( bn1, ( 2, 2 ),( 2, 2 ),name='pool1')

    conv2_1 = tf.layers.conv2d( pooling1,128,( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv2_1')
    #conv2_2 = tf.layers.conv2d( conv2_1,128,( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv2_2')
    bn2 = batch_norm_layer(conv2_1,is_training=use_bn,name='batch_norm2_1')
    pooling2 = tf.layers.max_pooling2d( bn2,( 2, 2 ),( 2, 2 ), name='pool2')

    conv3_1 = tf.layers.conv2d( pooling2,256,  ( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv3_1')
    #conv3_2 = tf.layers.conv2d( conv3_1,256, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv3_2')
    #conv3_3 = tf.layers.conv2d( conv3_2,256, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv3_3')
    bn3 = batch_norm_layer(conv3_1,is_training=use_bn,name='batch_norm3_1')
    pooling3 = tf.layers.max_pooling2d( bn3,( 2, 2 ),( 2, 2 ),name='pool3')
        
    conv4_1 = tf.layers.conv2d( pooling3,512,  ( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv4_1')
    #conv4_2 = tf.layers.conv2d( conv4_1,512, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv4_2')
    #conv4_3 = tf.layers.conv2d( conv4_2,512, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv4_3')
    bn4 = batch_norm_layer(conv4_1,is_training=use_bn,name='batch_norm4_1')
    pooling4 = tf.layers.max_pooling2d( bn4,( 2, 2 ),( 2, 2 ),name='pool4')
    
    conv5_1 = tf.layers.conv2d( pooling4,512,  ( 3, 3 ), padding = 'same',activation = tf.nn.relu,name = 'conv5_1')
    bn5 = batch_norm_layer(conv5_1,is_training=use_bn,name='batch_norm5_1')
    #conv5_2 = tf.layers.conv2d( conv5_1,512, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv5_2')
    #conv5_3 = tf.layers.conv2d( conv5_2,512, ( 3, 3 ),padding = 'same',activation = tf.nn.relu,name = 'conv5_3')
    #pooling5 = tf.layers.max_pooling2d( conv5_2,( 2, 2 ),( 2, 2 ),name='pool5')
    # 展平
    flatten  = tf.contrib.layers.flatten(bn5)
    y_ = tf.layers.dense(flatten, 10)

    return y_
