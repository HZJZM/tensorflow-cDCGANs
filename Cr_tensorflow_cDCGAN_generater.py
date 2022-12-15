import os, time, itertools, imageio, pickle, random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
from data_loader import DataLoader2
import hdf5storage


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# leaky_relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


# G(z)
def generator(x, z_fill, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer
        # cat1 = tf.concat([x, z_fill], 1)
        cat1 = x
        # cat1 = y_label
        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 128, [3, 3], strides=(2, 2), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init)
        # clrelu1 = lrelu(conv1, 0.2)
        clrelu1 = tf.nn.max_pool(conv1,[1,3,3,1],[1,2,2,1],padding='VALID')

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(clrelu1, 32, [3, 3], strides=(2, 2), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init)

        # clrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        # clrelu2 = tf.nn.max_pool(conv2,[1,3,3,1],[1,2,2,1],padding='VALID')
        clrelu2 = conv2

        conv3 = tf.layers.conv2d(clrelu2, 8, [3, 3], strides=(2, 1), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init)

        # clrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
        # clrelu3 = tf.nn.max_pool(conv3,[1,3,3,1],[1,1,1,1],padding='VALID')
        flattened = tf.reshape(conv3, [-1, 1 * 1 * 392])
        fc4 = fc(flattened, 1 * 1 * 392, 1*1*128, name='fc4')

        rs4 = tf.reshape(fc4,[-1,1,1,128])
        # conv4 = tf.layers.conv2d(clrelu3, 1, [3, 3], strides=(2, 1), padding='same', kernel_initializer=w_init,
        #                          bias_initializer=b_init)
        # clrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        # clrelu4 = tf.nn.max_pool(conv4,[1,3,3,1],[1,2,2,1],padding='VALID')


        # conv5 = tf.layers.conv2d(clrelu4, 32, [3, 3], strides=(7, 7), padding='same', kernel_initializer=w_init,
        #                          bias_initializer=b_init)
        # clrelu5 = lrelu(tf.layers.batch_normalization(conv5, training=isTrain), 0.2)
        # clrelu5 = lrelu(tf.layers.batch_normalization(rs4, training=isTrain), 0.2)

        clrelu5 = tf.concat([rs4, z_fill], 3)

        #
        deconv1 = tf.layers.conv2d_transpose(clrelu5, 256, [7, 7], strides=(1, 1), padding='valid', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(tf.layers.batch_normalization(deconv1, training=isTrain), 0.2)

        # 2nd hidden layer
        deconv2 = tf.layers.conv2d_transpose(lrelu1, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(deconv2, training=isTrain), 0.2)

        deconv3 = tf.layers.conv2d_transpose(lrelu2, 64, [5, 5], strides=(2, 2), padding='same',
                                             kernel_initializer=w_init, bias_initializer=b_init)
        lrelu3 = lrelu(tf.layers.batch_normalization(deconv3, training=isTrain), 0.2)

        # output layer
        deconv4 = tf.layers.conv2d_transpose(lrelu3, 3, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        o = tf.nn.tanh(deconv4)
        # o1 = tf.concat([y_fill,o],axis=1)
        return o

# D(x)
def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer
        # cat1 = tf.concat([x, y_fill], 1)
        cat1 = x

        # 1st hidden layer
        conv1 = tf.layers.conv2d(cat1, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [5, 5], strides=(2, 1), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        conv3 = tf.layers.conv2d(lrelu2, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init)
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        conv4 = tf.layers.conv2d(lrelu3, 256, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init)
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv2d(lrelu4, 1, [7, 7], strides=(1, 1), padding='valid', kernel_initializer=w_init)
        o = tf.nn.sigmoid(conv5)

        return o, conv5
        # output layer
        # conv4 = tf.layers.conv2d(lrelu3, 1, [5, 7], strides=(1, 1), padding='valid', kernel_initializer=w_init)
        # o = tf.nn.sigmoid(conv4)
        #
        # return o, conv4

def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights'+name, shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases'+name, [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


# preprocess
img_size = 56
# temp_z_ = np.random.normal(0, 1, (10, 1, 1, 100))
# fixed_z_ = temp_z_
# fixed_y_ = np.zeros((10, 1))
# for i in range(9):
#     fixed_z_ = np.concatenate([fixed_z_, temp_z_], 0)
#     temp = np.ones((10, 1)) + i
#     fixed_y_ = np.concatenate([fixed_y_, temp], 0)

# fixed_y_ = onehot[fixed_y_.astype(np.int32)].reshape((100, 1, 1, 21))
def gen_img(mat_file):
    all_data = hdf5storage.loadmat(mat_file)
    all_img_z = all_data['train_data']/255
    all_lab = all_data['label_data']
    gen_mat = np.zeros((1,56,56,3))
    lab_list = [] #labels shu liang tiao zheng zhi hou
    #数量调整
    for i in np.unique(all_lab):

    # for iter_d in range(all_img_z.shape[0] // batch_size): #single nd generate
        # update discriminator
        # batch_z = all_img_z[iter_d * batch_size:(iter_d + 1) * batch_size]
        batch_z = all_img_z[np.where(all_lab==i)]
        while np.shape(batch_z)[0]<100:
            batch_z = np.concatenate((batch_z,batch_z))
        batch_z = batch_z[np.random.randint(0,np.shape(batch_z)[0],100)]
        lab_list.extend([i]*100)
        batch_size = 100
    # z_fill_ = np.random.normal(loc=0, scale=0.1, size=(batch_size, img_size, img_size, 3))
        z_fill_ = np.random.normal(0, 1, (batch_size, 1, 1, 264))

        gen_images = sess.run(G_z, {z: batch_z, z_fill:z_fill_ , isTrain: False})
        gen_mat = np.concatenate((gen_mat,gen_images),axis=0)
    gen_mat = gen_mat[2:]
    nongdu_list = []
    for j in range(np.shape(gen_mat)[0]):
        nongdu = lab_list[j]
        nongdu_list.append(nongdu)
        cv2.imwrite('./gen_hzj812/'+str(lab_list[j])+'-'+str(nongdu_list.count(nongdu))+'.png', gen_mat[j]*255)

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training parameters
# batch_size = 64
# lr = 0.0002

# lr = tf.trai32n.exponential_decay(0.0002, global_step, 50, 0.95, staircase=True)

# load MNIST
# mnist = input_data.read_data_sets("data/", one_hot=True, reshape=[])
# dataloader.pre_process()
# dataloader.Data_Aug(10,56)
# variables : input
x = tf.placeholder(tf.float32, shape=(None, img_size*2, img_size, 3))
z = tf.placeholder(tf.float32, shape=(None, img_size*2, img_size, 3))
# y_label = tf.placeholder(tf.float32, shape=(None, 1, 1, 21))
# z_fill = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 3))
z_fill = tf.placeholder(tf.float32, shape=(None, 1, 1, 264))

y_fill = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 3))

isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, z_fill, isTrain)

# open session and initialize all variables
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
saver = tf.train.Saver()
saver.restore(sess,'./model/my-model-40')




np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()

gen_img('tensorflow-cDCGAN/hzj812_train_2to1for2_1.1.mat')


end_time = time.time()
total_ptime = end_time - start_time



sess.close()

