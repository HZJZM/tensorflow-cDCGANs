#resize为56x56的，并用mat文件保存图像和标签
import numpy as np
import os
from scipy.io import loadmat, savemat
import random
from scipy import stats
import re
import hdf5storage
from decimal import Decimal

def make_mat(data_path):
    import cv2
    all_img = None
    all_nd = []
    for i, j, k in os.walk(data_path):
        # print(i,j,k)
        for file in k:
            abs_path = os.path.join(i,file)
            pattern = re.compile('\d+(\.\d{1,2})?\-')
            nongdu = float(pattern.search(file).group()[:-1])
            # nongdu = float(file.strip('.jpg').split('-')[0])
            img = cv2.imread(abs_path)
            img_shape = np.shape(img)
            print(img_shape)
            # center = img.shape / 2
            x = img_shape[1]/2 - 56 / 2
            y = img_shape[0]/2 - 56 / 2

            img = img[int(y):int(y + 56), int(x):int(x + 56)]
            # img = cv2.resize(img,(56,56))
            # cv2.imwrite('tensorflow-cDCGAN/220521CR5656/a.png', img)
            # img = img[(img_shape[0]//2-150):(img_shape[0]//2+150),(img_shape[1]//2-150):(img_shape[1]//2+150),:]
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
    savemat('tensorflow-cDCGAN/220521CR_1-4_for2.mat',{'feature':all_img,'label':all_nd})

def make_mat21(matafile_path):
    tr_data = np.zeros((1,56*2,56,3))
    lb_data = np.zeros((1,56*2,56,3))
    all_data = loadmat(matafile_path)
    all_feature = all_data['feature']
    all_label = all_data['label']
    unq_lab = np.unique(all_label)
    for i in range(np.shape(unq_lab)[0]-1):
        chs_lab = unq_lab[i]
        for j in [ 0.1, 0.2, 0.3]:
            if not unq_lab.__contains__(float(Decimal(str(chs_lab))-Decimal(str(j)))) or not unq_lab.__contains__(float(Decimal(str(chs_lab))+Decimal(str(j)))):
                break
            mid_nd = np.where(all_label==(chs_lab))
            low_nd = np.where(all_label==(float(Decimal(str(chs_lab))-Decimal(str(j)))))
            hig_nd = np.where(all_label==(float(Decimal(str(chs_lab))+Decimal(str(j)))))

            try: tr_data = np.vstack((tr_data,np.concatenate((all_feature[np.random.choice(low_nd[1],64)],all_feature[np.random.choice(hig_nd[1],64)]), axis=1)))
            except:
                print('x')
            # lb_data_tmp = np.concatenate((all_feature[np.random.choice(mid_nd[1],1)],all_feature[np.random.choice(mid_nd[1],64)]), axis=1)
            lb_data_tmp = np.concatenate(
                (np.tile(all_feature[np.random.choice(mid_nd[1], 1)], (64, 1, 1, 1)),
                 all_feature[np.random.choice(mid_nd[1], 64)]), axis=1)
            lb_data = np.vstack((lb_data, lb_data_tmp))
    # savemat('tensorflow-cDCGAN/220521_2-1.mat',
    #                      {'train_data': tr_data[1:], 'label_data': lb_data[1:]})
    hdf5storage.savemat('tensorflow-cDCGAN/hzj812_train_2to1.mat', {'train_data': tr_data[1:], 'label_data': lb_data[1:]}, format='7.3', matlab_compatible=True)

def make_mat21_for2(matafile_path):
    tr_data = np.zeros((1,56*2,56,3))
    lb_data = []
    all_data = loadmat(matafile_path)
    all_feature = all_data['feature']
    all_label = all_data['label']
    unq_lab = np.unique(all_label)
    for i in range(1,30,1):
        chs_lab = i/10
        for j in [0.1, 0.2, 0.3]:
            if not unq_lab.__contains__(float(Decimal(str(chs_lab))-Decimal(str(j)))) or not unq_lab.__contains__(float(Decimal(str(chs_lab))+Decimal(str(j)))):
                break
            # mid_nd = np.where(all_label==(chs_lab))
            low_nd = np.where(all_label==(float(Decimal(str(chs_lab))-Decimal(str(j)))))
            hig_nd = np.where(all_label==(float(Decimal(str(chs_lab))+Decimal(str(j)))))

            try: tr_data = np.vstack((tr_data,np.concatenate((all_feature[np.random.choice(low_nd[1],64)],all_feature[np.random.choice(hig_nd[1],64)]), axis=1)))
            except:
                print('x')
            # lb_data_tmp = np.concatenate((all_feature[np.random.choice(mid_nd[1],64)],all_feature[np.random.choice(mid_nd[1],64)]), axis=1)
            lb_data_tmp = [chs_lab]*64
            lb_data.extend(lb_data_tmp)
    # savemat('tensorflow-cDCGAN/220521_2-1.mat',
    #                      {'train_data': tr_data[1:], 'label_data': lb_data[1:]})
    lb_data = np.array(lb_data)
    hdf5storage.savemat('tensorflow-cDCGAN/hzj812_train_2to1for2_1.1.mat', {'train_data': tr_data[1:], 'label_data': lb_data}, format='7.3', matlab_compatible=True)


# make_mat21('./Downloads/化学数据/hzj812_train_1.1.mat')
# make_mat('tensorflow-cDCGAN/220521R_1-4-for2_')
make_mat21_for2('./Downloads/化学数据/hzj812_train_1.1.mat')
# ./Downloads/化学数据/hzj812_train.mat
