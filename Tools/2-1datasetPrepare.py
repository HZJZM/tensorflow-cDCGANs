import os
import random

import cv2
from PIL import Image

import numpy as np
from scipy.io import savemat
import tensorflow as tf
import re

# file_path = '/220521CR__'
# gen_num = 200000
# img_size = 56
# tr_data = np.zeros((gen_num,img_size*2,img_size,3))
# lb_data = np.zeros((gen_num,img_size,img_size,3))

def date_ext(img, ext_num, mode=Image.BICUBIC):
    i=0
    imglist = [normal_img(img)]
    while i<ext_num:
        random_angle = np.random.randint(1,360)
        # img.rotate(random_angle, mode).show()
        ext_img = normal_img(img.rotate(random_angle, mode))
        imglist.append(ext_img)
        i+=1
    return imglist

def img_cut(file_path):
    tgt_path = './Downloads/化学数据/六价铬1月2日_cut' #target dir for save the new imgae
    if not os.path.exists(tgt_path):
        os.mkdir(tgt_path)
    for file in os.listdir(file_path):
        newfile = os.path.join(tgt_path,file) #new file path
        h1,h2,w1,w2 = 0,0,0,0
        skip_r = 50  #unit scan range
        var_limit = 300 #min value of the cut line
        abs_file = os.path.join(file_path,file) #original file abs path
        img = cv2.imread(abs_file)
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))#cover to PIL

        # cv2.namedWindow('111', 0)
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.namedWindow('222', 0)
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # h,w,c = np.shape(img)
        # for htmp in range(0,h,skip_r):
        #     var_cur = np.var(img[int(htmp), :, 0])
        #     if var_cur > var_limit:
        #         h1 = htmp
        #         break
        # for htmp in range(0, h, skip_r):
        #     var_cur = np.var(img[-int(htmp), :, 0])
        #     if var_cur > var_limit:
        #         h2 = h - htmp
        #         break
        # for wtmp in range(0, w, skip_r):
        #     var_cur = np.var(img[:, int(wtmp), 0])
        #     if var_cur > var_limit:
        #         w1 = wtmp
        #         break
        # for wtmp in range(0, w, skip_r):
        #     var_cur = np.var(img[:, -int(wtmp), 0])
        #     if var_cur > var_limit:
        #         w2 = w - wtmp
        #         break
        # print(h1, h2, w1, w2)
        # image = img[h1:h2,w1:w2]
        # cv2.namedWindow('111', 0)
        # cv2.imshow("image", image)
      #  cv2.waitKey(0)
      #   cv2.destroyAllWindows()
        cv2.imwrite(newfile,img)

def normal_img(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    width, hight = np.shape(img)[0],np.shape(img)[1]
    new_lenght = min(width,hight)-100
    bia_w, bia_r = int((width-new_lenght)//2), int((hight-new_lenght)//2) #bia of top-blow right-left
    new_img = cv2.resize(img[bia_w:bia_w+new_lenght,bia_r:bia_r+new_lenght],(56,56))
    # cv2.namedWindow('111', 0)
    # cv2.imshow("image", new_img)
    # cv2.waitKey(0)
    return new_img

def make_mat1(data_path):
    all_img = None
    all_nd = []
    for i, j, k in os.walk(data_path):
        # print(i,j,k)
        for file in k:
            abs_path = os.path.join(i, file)
            pattern = re.compile('\d+(\.\d{1,2})?\-')
            nongdu = float(pattern.search(file).group()[:-1])
            img = cv2.imread(abs_path)
            image = Image.fromarray(img)  # cover to PIL
            extlist = date_ext(image, 5)
            if all_img is None:
                all_img = np.array(extlist)
            else:
                print(file)
                all_img = np.vstack((all_img, np.array(extlist)))
            # all_img.append(img.tolist())
            all_nd.extend([nongdu]*len(extlist))
    savemat('tensorflow-cDCGAN/六价铬1月2日_cut56x56.mat',
                {'feature': all_img, 'label': all_nd})


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
            img = normal_img(img)
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


# img_cut('./Downloads/化学数据/六价铬1月2日')

make_mat1('./Downloads/化学数据/六价铬1月2日_cut')