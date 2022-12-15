import os
import re
import shutil

import cv2
import numpy as np
from scipy.io import savemat

sour_path = '../220521R_1-4'
tgt_path = '../220521R_1-4-for2'
all_img = None
all_nd = []
files_sou = os.listdir(sour_path)
for file in files_sou:
    pattern = re.compile('\d+(\.\d{1,2})?\-')
    nongdu = float(pattern.search(file).group()[:-1])
    if nongdu*10%2 == 0:
        file_path = os.path.join(sour_path,file)
        img = cv2.imread(file_path)
        if all_img is None:
            all_img = img
        else:
            print(file)
            all_img = np.vstack((all_img, img))
            # all_img.append(img.tolist())
        all_nd.append(nongdu)
        shutil.copy(file_path,os.path.join(tgt_path,file))
all_nd = np.array(all_nd)
savemat('tensorflow-cDCGAN/220521CR_for2.mat',{'feature':all_img,'label':all_nd})
