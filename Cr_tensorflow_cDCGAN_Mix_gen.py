import os, time, itertools, imageio, pickle, random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
from data_loader import DataLoaderMix
# from keras.utils import to_categorical


# leaky_relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)



# G(z)
def generator(x, z_fill,isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer
        cat1 = tf.concat([x, z_fill], 3)
        # cat1 = x
        # cat1 = y_label
        # 1st hidden layer
        # 1st hidden layer
        conv1 = tf.layers.conv2d(cat1, 128, [1, 9], strides=(2, 2), padding='valid', kernel_initializer=w_init,
                                 bias_initializer=b_init)
        lrelu1 = lrelu(conv1, 0.2)

        deconv1 = tf.layers.conv2d_transpose(lrelu1, 256, [5, 5], strides=(1, 1), padding='valid',
                                             kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(tf.layers.batch_normalization(deconv1, training=isTrain), 0.2)

        # 2nd hidden layer
        deconv2 = tf.layers.conv2d_transpose(lrelu1, 128, [5, 5], strides=(2, 2), padding='same',
                                             kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(deconv2, training=isTrain), 0.2)

        # output layer
        deconv3 = tf.layers.conv2d_transpose(lrelu2, 1, [5, 5], strides=(2, 2), padding='same',
                                             kernel_initializer=w_init, bias_initializer=b_init)
        o = tf.nn.tanh(deconv3)

        return o


# D(x)
def discriminator(x, y_fill, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer
        cat1 = tf.concat([x, y_fill], 3)

        # 1st hidden layer
        conv1 = tf.layers.conv2d(cat1, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # output layer
        conv3 = tf.layers.conv2d(lrelu2, 1, [5, 5], strides=(1, 1), padding='valid', kernel_initializer=w_init)
        o = tf.nn.sigmoid(conv3)

        return o, conv3
# preprocess
img_size = 20
onehot = np.eye(21)
# temp_z_ = np.random.normal(0, 1, (10, 1, 1, 100))
# fixed_z_ = temp_z_
# fixed_y_ = np.zeros((10, 1))
# for i in range(9):
#     fixed_z_ = np.concatenate([fixed_z_, temp_z_], 0)
#     temp = np.ones((10, 1)) + i
#     fixed_y_ = np.concatenate([fixed_y_, temp], 0)

# fixed_y_ = onehot[fixed_y_.astype(np.int32)].reshape((100, 1, 1, 21))
def show_result(test_label, _range, fea_min, num_epoch, show = False, save = True, path = 'result.png', ):
    z_ = np.random.normal(0, 1, (2, 1, 9, 100))
    # y_ = np.random.randint(0, 10, (batch_size, 9))
    y_ = np.random.randint(0, 10, (2, 9))
    # z_fill_ = np.zeros((batch_size, 1, 9, 10))
    # for i in range(batch_size):
    #     z_fill_[i][0] = np.eye(10)[y_[i]]

    z_fill_ = test_label
    i=0
    for i in range(np.shape(y_)[0]):
        y_[i] = [np.argmax(oneho) for oneho in z_fill_[i,0]]
    # z_fill_ = np.random.normal(loc=0, scale=0.1, size=(batch_size, img_size, img_size, 3))
    test_images = sess.run(G_z, {z: z_, z_fill:z_fill_ , isTrain: False})
    # test_images = np.clip(test_images,0,1)
    # for i in range(0,30,5):
    #     cv2.imwrite('./test2-1/'+str(num_epoch)+'-'+str(i)+'.png',test_images[i,112:168,:,:]*255)
    np.savetxt('./generater_sample_lab.txt',y_)
    np.savetxt('./generater_sample_fea.txt',test_images.reshape(2,400)*_range+fea_min,fmt='%.4f')


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
batch_size = 5
# lr = 0.0002
train_epoch = 150
global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(0.0003, global_step, 50, 0.75, staircase=True)
# lr = tf.train.exponential_decay(0.0002, global_step, 50, 0.95, staircase=True)

# load MNIST
# mnist = input_data.read_data_sets("data/", one_hot=True, reshape=[])
dataloader = DataLoaderMix('混合数据处理/mix_data.mat','混合数据处理/single_data.mat')
dataloader.fea_reshape()
dataloader.split_train_test()
data_range = dataloader._range
data_min = dataloader.feature_min
# dataloader.pre_process()
# dataloader.Data_Aug(10,56)
# variables : input
x = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 9, 100))
# z = np.random.normal(0, 1, (batch_size, 1, 9, 100))
# y_label = tf.placeholder(tf.float32, shape=(None, 1, 1, 9))
y_fill = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 10))
z_fill = tf.placeholder(tf.float32, shape=(None, 1, 9, 10))
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, z_fill, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, y_fill, isTrain)
D_fake, D_fake_logits = discriminator(G_z, y_fill, isTrain, reuse=tf.AUTO_REUSE)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
# mean1, variance1 = tf.nn.moments(G_z[:,50:250,50:250,:], 1)
# mean2, variance2 = tf.nn.moments(G_z[:,50:250,50:250,:], 2)
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))


# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optim = tf.train.AdamOptimizer(lr, beta1=0.5)
    D_optim = optim.minimize(D_loss, global_step=global_step, var_list=D_vars)
    # D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars,global_step=global_step)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

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
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    shuffle_idxs = random.sample(range(0, train_set.shape[0]), train_set.shape[0])
    shuffled_set = train_set[shuffle_idxs]
    shuffled_label = train_label[shuffle_idxs]
    for iter in range(shuffled_set.shape[0] // batch_size):
        # update discriminator
        label_ = shuffled_label[iter * batch_size:(iter + 1) * batch_size]
        data_ = shuffled_set[iter*batch_size:(iter+1)*batch_size]
        # x_ = np.concatenate((data_,label_),axis=1)
        # y_label_ = np.zeros((batch_size,1,1,21))
        # for i in range(batch_size):
        #     idx = int(label_[i]//1)
        #     y_label_[i,0,0,idx] = label_[i]
        # y_label_ = shuffled_label[iter*batch_size:(iter+1)*batch_size].reshape([batch_size, 1, 1, 21])
        # y_fill_ = y_label_ * np.ones([batch_size, img_size, img_size, 21])
        # y_fill_ = label_
        # z_fill_ = np.random.normal(loc=0, scale=0.1, size=(batch_size,1,9, 10))
        z_fill_ = label_
        # y_fill_ = z_fill_ * np.ones([batch_size, img_size, img_size, 10])
        y_fill_ = np.resize(z_fill_,(batch_size,20,20,10))
        z_ = np.random.normal(0, 1, (batch_size, 1, 9, 100))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: data_, z: z_, z_fill:z_fill_, y_fill:y_fill_, isTrain: True})

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 1, 9, 100))
        y_ = np.random.randint(0, 10, (batch_size, 9))
        z_fill_ = np.zeros((batch_size,1,9,10))
        for i in range(batch_size):
            z_fill_[i][0] = np.eye(10)[y_[i]]
        # y_fill_ = z_fill_ * np.ones([batch_size, img_size, img_size, 10])
        y_fill_ = np.resize(z_fill_,(batch_size,20,20,10))

        # z_ = data_
        # y_ = np.random.randint(0, 21, (batch_size, 1))
        # y_ = np.random.uniform(0, 20, batch_size)
        # y_ = np.around(y_, decimals=1)
        # y_ = np.random.randint(0,41,batch_size)/2
        # y_label_ = np.zeros((batch_size, 1, 1, 21))
        # for i in range(batch_size):
        #     idx = int(y_[i] // 1)
        #     y_label_[i, 0, 0, idx] = y_[i]
        # y_label_ = onehot[y_.astype(np.int32)].reshape([batch_size, 1, 1, 21])
        # y_fill_ = label_
        # z_fill_ = np.random.normal(loc=0, scale=0.1, size=(batch_size, img_size, img_size, 3))

        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: data_, z_fill:z_fill_, y_fill:y_fill_, isTrain: True})

        errD_fake = D_loss_fake.eval({z: z_, z_fill:z_fill_, y_fill: y_fill_, isTrain: False})
        errD_real = D_loss_real.eval({x: data_, z_fill:z_fill_, y_fill: y_fill_, isTrain: False})
        errG = G_loss.eval({z: z_, z_fill:z_fill_, y_fill: y_fill_, isTrain: False})
        # errD_fake = D_loss_fake.eval({z: z_,  z_fill:z_fill_ , isTrain: False})
        # errD_real = D_loss_real.eval({x: data_, z_fill:z_fill_ , isTrain: False})
        # errG = G_loss.eval({z: z_,  z_fill:z_fill_ , isTrain: False})

        D_losses.append(errD_fake + errD_real)
        G_losses.append(errG)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result(dataloader.label_test, data_range, data_min, (epoch + 1), save=True, path=fixed_p, )
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