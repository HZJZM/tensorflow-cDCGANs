#合并文件夹

import os
import shutil

sour_path = './Downloads/化学数据/六价铬1月2日/'
tgt_path = '/220521CR'

for file in os.listdir(sour_path):
    sour_file = os.path.join(sour_path,file)
    if os.path.exists(os.path.join(tgt_path,file)):
        file = file.replace('.png','-.png')
    tgt_file = os.path.join(tgt_path,file)
    shutil.copy(sour_file,tgt_file)