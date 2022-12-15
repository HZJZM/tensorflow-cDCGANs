import os, time, itertools, imageio, pickle, random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
from data_loader import DataLoader2

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
def show_result(num_epoch, show = False, save = True, path = 'result.png',y_fill_=None, show_set=None):
    # z_fill_ = np.random.normal(loc=0, scale=0.1, size=(batch_size, img_size, img_size, 3))
    z_fill_ = np.random.normal(0, 1, (batch_size, 1, 1, 264))

    test_images = sess.run(G_z, {z: show_set, z_fill:z_fill_ ,y_fill:y_fill_[:, :56, :, :],  isTrain: False})
    # test_images = np.clip(test_images,0,1)
    for i in range(0,30,5):
        show_img = np.concatenate((y_fill_[i, :56, :, :], test_images[i, :, :, :]), axis=1)
        cv2.imwrite('./test2-1/'+str(num_epoch)+'-'+str(i)+'.png', show_img*255)
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
batch_size = 128
# lr = 0.0002
train_epoch = 60
global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(0.0001, global_step, 100, 0.95, staircase=True)

# lr_G = tf.train.exponential_decay(0.00001, global_step, 1, 0.9, staircase=True)
# lr_D = tf.train.exponential_decay(0.0000005, global_step, 1, 0.9, staircase=True)

# lr = tf.trai32n.exponential_decay(0.0002, global_step, 50, 0.95, staircase=True)

# load MNIST
# mnist = input_data.read_data_sets("data/", one_hot=True, reshape=[])
dataloader = DataLoader2('tensorflow-cDCGAN/hzj812_train_2to1.mat')
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
G_z_conc = tf.concat([G_z,y_fill],axis=1)

# networks : discriminator
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z_conc, isTrain, reuse=tf.AUTO_REUSE)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
# mean1, variance1 = tf.nn.moments(G_z[:,50:250,50:250,:], 1)
# mean2, variance2 = tf.nn.moments(G_z[:,50:250,50:250,:], 2)
# mean, variance = tf.nn.moments(G_z,[1,2])
# G_loss_var = tf.reduce_sum(variance)
# G_z_split = tf.split(G_z,num_or_size_splits=2,axis=1)
G_loss_rec = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(tf.abs(G_z-y_fill), axis=[1,2]), reduction_indices=[1]))#reconstruct loss
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))+0.0001*G_loss_rec

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optim = tf.train.AdamOptimizer(lr, beta1=0.5)
    D_optim = optim.minimize(D_loss, global_step=global_step, var_list=D_vars)
    # D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)
    # optim_G = tf.train.AdamOptimizer(lr_G, beta1=0.5)
    # optim_D = tf.train.AdamOptimizer(lr_D, beta1=0.5)

    # D_optim = optim_D.minimize(D_loss, global_step=global_step, var_list=D_vars)
    # D_optim = tf.train.AdamOptimizer(lr_D, beta1=0.5).minimize(D_loss)
    # G_optim = tf.train.AdamOptimizer(lr_G, beta1=0.5).minimize(G_loss)

# open session and initialize all variables
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
saver = tf.train.Saver()
# saver.restore(sess,'./model/存起来/my-model-280')


# MNIST resize and normalization
# train_set = tf.image.resize_images(mnist.train.images, [img_size, img_size]).eval()
# train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1
# train_set = (mnist.train.images - 0.5) / 0.5
# train_label = mnist.train.labels
train_set = dataloader.all_feature
# train_set = (train_set-0.5)/0.5
train_label = dataloader.all_label
# train_label = np.eye(21)[np.array(train_label, dtype=np.int)]
# train_label = train_label[0]

# results save folder
root = 'cDCGAN_results/'
model = 'cDCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
G_epoch = 10
D_epoch = 5
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()


    for d_epoch in range(D_epoch):
        shuffle_idxs = random.sample(range(0, train_set.shape[0]), train_set.shape[0])
        shuffled_set = train_set[shuffle_idxs]
        shuffled_label = train_label[shuffle_idxs]
        loss_d = 0
        iter_d = 0
        for iter_d in range(shuffled_set.shape[0] // batch_size):
            # update discriminator
            label_ = shuffled_label[iter_d * batch_size:(iter_d + 1) * batch_size]
            data_ = shuffled_set[iter_d*batch_size:(iter_d+1)*batch_size]
            # x_ = np.concatenate((label_,label_),axis=1)
            x_ = label_
            # y_label_ = np.zeros((batch_size,1,1,21))
            # for i in range(batch_size):
            #     idx = int(label_[i]//1)
            #     y_label_[i,0,0,idx] = label_[i]
            # y_label_ = shuffled_label[iter*batch_size:(iter+1)*batch_size].reshape([batch_size, 1, 1, 21])
            # y_fill_ = y_label_ * np.ones([batch_size, img_size, img_size, 21])
            y_fill_ = label_[:,:56,:,:]
            # z_fill_ = np.random.normal(loc=0, scale=0.1, size=(batch_size,img_size,img_size, 3))
            z_fill_ = np.random.normal(0, 1, (batch_size, 1, 1, 264))

            loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: data_,z_fill:z_fill_,y_fill:y_fill_, isTrain: True})
            loss_d += loss_d_
        print('[%d/%d]  loss_d: %.3f' % ((epoch + 1), train_epoch, loss_d))
        D_losses.append(loss_d)
    for g_epoch in range(G_epoch):
        loss_g = 0
        iter_g = 0
        shuffle_idxs = random.sample(range(0, train_set.shape[0]), train_set.shape[0])
        shuffled_set = train_set[shuffle_idxs]
        shuffled_label = train_label[shuffle_idxs]
        for iter_g in range(shuffled_set.shape[0] // batch_size):
            # update discriminator
            label_ = shuffled_label[iter_g * batch_size:(iter_g + 1) * batch_size]
            data_ = shuffled_set[iter_g * batch_size:(iter_g + 1) * batch_size]
            # x_ = np.concatenate((label_,label_),axis=1)
            x_ = label_
            # y_label_ = np.zeros((batch_size,1,1,21))
            # for i in range(batch_size):
            #     idx = int(label_[i]//1)
            #     y_label_[i,0,0,idx] = label_[i]
            # y_label_ = shuffled_label[iter*batch_size:(iter+1)*batch_size].reshape([batch_size, 1, 1, 21])
            # y_fill_ = y_label_ * np.ones([batch_size, img_size, img_size, 21])
            y_fill_ = label_[:, :56, :, :]
            # z_fill_ = np.random.normal(loc=0, scale=0.1, size=(batch_size,img_size,img_size, 3))
            z_fill_ = np.random.normal(0, 1, (batch_size, 1, 1, 264))
            loss_g_, _ = sess.run([G_loss, G_optim],
                                  {z:data_, x: x_, z_fill: z_fill_, y_fill: y_fill_, isTrain: True})
            loss_g += loss_g_
        print('[%d/%d]  loss_g: %.3f' % ((epoch + 1), train_epoch, loss_g))
        G_losses.append(loss_g)
        # update generator
        # y_ = np.random.randint(0, 21, (batch_size, 1))
        # y_ = np.random.uniform(0, 20, batch_size)
        # y_ = np.around(y_, decimals=1)
        # y_ = np.random.randint(0,41,batch_size)/2
        # y_label_ = np.zeros((batch_size, 1, 1, 21))
        # for i in range(batch_size):
        #     idx = int(y_[i] // 1)
        #     y_label_[i, 0, 0, idx] = y_[i]
        # y_label_ = onehot[y_.astype(np.int32)].reshape([batch_size, 1, 1, 21])
        # y_fill_ = label_[:,:56,:,:]
        # z_fill_ = np.random.normal(loc=0, scale=0.1, size=(batch_size,img_size,img_size, 3))
        # z_fill_ = np.random.normal(0, 1, (batch_size, 1, 1, 64))

        # loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, z_fill:z_fill_,y_fill:y_fill_, isTrain: True})

        # errD_fake = D_loss_fake.eval({z: data_,  z_fill:z_fill_ , y_fill:y_fill_, isTrain: False})
        # errD_real = D_loss_real.eval({x: x_, z_fill:z_fill_ , y_fill:y_fill_, isTrain: False})
        # errG = G_loss.eval({z: data_,  z_fill:z_fill_ , y_fill:y_fill_, isTrain: False})
        #
        # D_losses.append(errD_fake + errD_real)
        # G_losses.append(errG)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    #测试高低浓度样本
    test_idx = np.random.choice(shuffle_idxs, batch_size)
    test_set = train_set[test_idx]
    test_fill = np.array(train_label[test_idx],dtype=np.float32)
    show_result((epoch + 1), save=True, path=fixed_p, y_fill_=test_fill, show_set=test_set)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    if epoch % 5 == 0:
        # 保存checkpoint, 同时也默认导出一个meta_graph
        # graph名为'my-model-{global_step}.meta'.
        saver.save(sess, './model/my-model', global_step=epoch)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

sess.close()
