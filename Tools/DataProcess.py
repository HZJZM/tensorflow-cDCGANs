#!/usr/bin/env Python
# coding=utf-8

import numpy as np
import os
from scipy.io import loadmat, savemat
import random
from scipy import stats
import re
import cv2
import matplotlib as plt

class DataLoader(object):
    def __init__(self,mat_file):
        all_data = loadmat(mat_file)
        self.all_feature = all_data['feature']
        self.all_label = all_data['label']

    def Porcess_norm(self):
        #标准化
        for i in range(self.all_feature.shape[0]):
            for j in range(3):
                f_mode = stats.mode(self.all_feature[i,:,:,j])[0][0][0]
                f_median = np.median(self.all_feature[i,:,:,j])
                f_mean = np.mean(self.all_feature[i,:,:,j])
                f_max = max(f_mode,f_median,f_mean)
                f_min = min(f_mode,f_median,f_mean)
                self.all_feature[i,:,:,j] = np.clip(self.all_feature[i,:,:,j],(f_min+f_max)/2-0.2,(f_min+f_max)/2+0.2)

    def Data_Aug(self, Aug_ratio, sub_size): #Data Augmentation by sagmentation
        tg_num = np.shape(self.all_feature)[0]*Aug_ratio
        self.all_label = np.repeat(self.all_label,[Aug_ratio])
        tmp_arr = np.zeros((tg_num,sub_size,sub_size,3))
        orig_size = np.shape(self.all_feature)[1] #original size
        rp1 = random.randint(0,orig_size-1-sub_size)#Random points
        rp2 = random.randint(0,orig_size-1-sub_size)
        rand_bia = random.randrange(-10,10)
        rp1 = orig_size//2+rand_bia
        for i in range(tg_num):
            idx = i//Aug_ratio
            tmp_arr[i,:] = self.all_feature[idx,rp1:rp1+sub_size, rp2:rp2+sub_size]
        self.all_feature = tmp_arr

    def get_batch(self, batch_size):
            """
            get a batch of images and corresponding labels.
            """
            #生成样本细致程度慢慢来
            self.all_label = self.all_label[np.where(self.all_label%1==0)]
            self.all_feature = self.all_feature[np.where(self.all_label%1==0)]
            num_samples = np.shape(self.all_feature)[0]

            idx = np.random.randint(num_samples, size=batch_size)

            return self.all_feature[idx], self.all_label[idx]

# A = DataLoader('files/Cr.mat')
# A.Data_Aug(25,28)
def cut_img(img):
    loc = np.where(img[:,:,1]>65)
    ind, times = np.unique(loc[0],return_counts=True,axis=0) # range of x
    left = min(ind[np.where(times>80)])
    right = max(ind[np.where(times>80)])
    ind_,times_ = np.unique(loc[1],return_counts=True,axis=0)
    down = min(ind_[np.where(times_>120)])
    top = max(ind_[np.where(times_>120)])
    #range of y
    # y_idx_l = np.min(np.where(loc[0]==left))
    # y_idx_r = np.max(np.where(loc[0]==right))
    # y_idx = loc[1][y_idx_l:y_idx_r]
    # ind_,times_ = np.unique(loc[])
    # top1 = max(loc[1][np.where(loc[0]==left)])
    # down1 = min(loc[1][np.where(loc[0]==left)])
    # top2 = max(loc[1][np.where(loc[0]==right)])
    # down2 =min(loc[1][np.where(loc[0]==right)])
    # top = max([top1,top2])
    # down = min([down1, down2])
    # top = int(np.mean(y_idx)) + 200
    # down = int(np.mean(y_idx)) - 200
    # righ_down = [max(loc[0]),max(loc[1])]
    return img[left:right,down:top,:]

def prepare_data(data_path):
    import cv2
    all_img = None
    all_nd = []
    for i, j, k in os.walk(data_path):
        # print(i,j,k)
        for file in k:
            if file.startswith('beijing'):
                continue
            abs_path = os.path.join(i,file)
            # nongdu = float(file.strip('.jpg').split('-')[0])
            pattern = re.compile('\d+(\.\d{1,2,3})?\-')
            nongdu = float(pattern.search(file).group()[:-1])
            fpath = data_path+'_/'
            # if 0 == 0:
            img = cv2.imread(abs_path)
            img = cut_img(img)

            # continue cut
            pointx, pointy = np.shape(img)[0]//2-200, np.shape(img)[1]//2-200
            for c in range(3):
                cut_bias1 = random.randint(0,30)
                cut_bias2 = random.randint(0,30)

                pointx_,pointy_ = pointx+cut_bias1, pointy+cut_bias2
                img_rand = img[pointx_:pointx_+400,pointy_:pointy_+400,:]
                hor = cv2.flip(img_rand, 1)
                # 垂直翻转
                ver = cv2.flip(img_rand, 0)
                # 水平垂直翻转
                hor_ver = cv2.flip(img_rand, -1)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                cv2.imwrite(fpath+file.strip('.png') +'_'+str(c)+'_.png',img_rand)
                cv2.imwrite(fpath+file.strip('.png')+'_'+str(c)+'hor'+'_.png',hor)
                cv2.imwrite(fpath+file.strip('.png')+'_'+str(c)+'ver'+'_.png',ver)
                cv2.imwrite(fpath+file.strip('.png')+'_'+str(c)+'hor_ver'+'_.png',hor_ver)


                # cv2.imwrite(data_path + '_/' + file, img)
            # img = img[(img_shape[0]//2-80):(img_shape[0]//2+80),(img_shape[0]//2-80):(img_shape[0]//2+80),:]
            # img = img[np.newaxis,:]
            # if all_img is None:
            #     all_img = img
            # else:
            #     all_img = np.vstack((all_img,img))
            # all_img.append(img.tolist())
            # all_nd.append(nongdu)
            # np.array(all_img)
    # all_nd = np.array(all_nd)
    # savemat('files/Cr.mat',{'feature':all_img,'label':all_nd})

def make_mat(data_path):
    import cv2
    all_img = None
    all_nd = []
    for i, j, k in os.walk(data_path):
        # print(i,j,k)
        for file in k:
            abs_path = os.path.join(i,file)
            nongdu = float(file.strip('.jpg').split('-')[0])
            img = cv2.imread(abs_path)
            img_shape = np.shape(img)
            print(img_shape)
            img = img[(img_shape[0]//2-150):(img_shape[0]//2+150),(img_shape[1]//2-150):(img_shape[1]//2+150),:]
            img = img[np.newaxis,:]
            if all_img is None:
                all_img = img
            else:
                print(file)
                all_img = np.vstack((all_img,img))
            # all_img.append(img.tolist())
            all_nd.append(nongdu)
            # np.array(all_img)
    all_nd = np.array(all_nd)
    savemat('files/Cr_20nd.mat',{'feature':all_img,'label':all_nd})


def make_mat812(data_path):
    import cv2
    all_img = None
    all_nd = []
    for i, j, k in os.walk(data_path):
        # print(i,j,k)
        if not os.path.exists(data_path + '_/'):
            os.makedirs(data_path + '_/')
        for file in k:
            for c in range(10):
                abs_path = os.path.join(i,file)
                nongdu = float(file.strip('.jpg').split('-')[0])
                img = cv2.imread(abs_path)
                img_shape = np.shape(img)
                print(img_shape)
                pointx = random.randint(1,56)
                pointy = random.randint(1,56)
                img = img[(img_shape[0]//2-pointx):(img_shape[0]//2-pointx+56),(img_shape[1]//2-pointy):(img_shape[1]//2-pointy+56),:]
                # img = np.resize(img,(56,56,3))
                # img = cv2.resize(img, (56, 56), interpolation=cv2.INTER_CUBIC)

                cv2.imwrite(data_path + '_/' + file.strip('.png') + '_' + str(c) + '_.png', img)
    #
    #             img = img[np.newaxis,:]
    #             if all_img is None:
    #                 all_img = img
    #             else:
    #                 print(file)
    #                 all_img = np.vstack((all_img,img))
    #
    #             # all_img.append(img.tolist())
    #             all_nd.append(nongdu)
    #         # np.array(all_img)
    # all_nd = np.array(all_nd)
    # savemat('./Downloads/化学数据/hzj812_train_1.1.mat',{'feature':all_img,'label':all_nd})

def togray(img_dir):
    fpath = img_dir + '_gray'
    if not os.path.isdir(fpath):
        os.makedirs(fpath)
    for f in os.listdir(img_dir):
        if f.endswith('.txt'):
            continue
        img = cv2.imread(os.path.join(img_dir,f))
        imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(fpath,f), imggray)

# togray('./Downloads/化学数据/hzj0812_test__')

# img = '/media/l/6069-F139/ZZH/0914-cys/0.5-2.png'
# togray(img)
# make_mat812('/media/l/6069-F139/ZZH/0914-Ca_test_')
#

#
prepare_data('/media/l/6069-F139/ZZH/0914-Ca_test')
# make_mat('./gittest/TensorFlow-Convolutional-AutoEncoder/铬')