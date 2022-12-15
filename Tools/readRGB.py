#!/usr/bin/env Python
# coding=utf-8
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
# Modules settings
import numpy as np
def cunt(dir_path):
    if not os.path.exists(dir_path+'_xy'):
        os.makedirs(dir_path+'_xy')
    with open(os.path.dirname(dir_path)+'/cunt.txt',"w+") as fp:
        for f in os.listdir(dir_path):
            if f.endswith('.txt'):
                continue
            wr_str = ""
            img = cv2.imread(os.path.join(dir_path,f))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                R_ind0,R_cunt0 = np.unique(img[:,:,0], return_counts=True)
            except:
                print(f)
                continue
            R_ind1, R_cunt1 = np.unique(img[:,:,1], return_counts=True)
            R_ind2, R_cunt2 = np.unique(img[:,:,2], return_counts=True)

            for i in range(len(R_ind0)):
                if R_cunt0[i]<100:
                    img[:,:,0][np.where(img[:,:,0]==R_ind0[i])] = 0
                    img[:,:,1][np.where(img[:,:,1]==R_ind0[i])] = 0
                    img[:,:,2][np.where(img[:,:,2]==R_ind0[i])] = 0


            try:
                R_x1 = np.reshape(img[:,:,0], (1,-1))
            except:
                print(f)
                continue
            G_y1 = np.reshape(img[:,:,1], (1,-1))

            R_x2 = np.reshape(img[:,:,1], (1,-1))
            G_y2 = np.reshape(img[:,:,2], (1,-1))

            R_x3 = np.reshape(img[:,:,0], (1,-1))
            G_y3 = np.reshape(img[:,:,2], (1,-1))
            # G_ind, G_cunt = np.unique(img[:,:,1], return_counts=True)

            plt.xlim((40, 130))
            plt.ylim((40, 130))
            plt.axis('off')
            plt.scatter(R_x1, G_y1,s=3,c='c')
            plt.scatter(R_x2, G_y2,s=3,c='m')
            plt.scatter(R_x3, G_y3,s=3,c='y')

            # plt.show()
            plt.savefig(os.path.join(dir_path+'_xy',f),dpi=15,bbox_inches = 'tight')
            plt.clf()
            # B_ind, B_cunt = np.unique(img[:,:,2], return_counts=True)
            # wr_str = 'RRR:'+str(R_ind.tolist())+'\n'+"RRR:"+str(R_cunt.tolist())+'\n'+"GGG:"+str(G_ind.tolist())+'\n'+"GGG:"+str(G_cunt.tolist())+'\n'+"BBB:"+str(B_ind.tolist())+'\n'+"BBB:"+str(B_cunt.tolist())+'\n'
            # fp.write(wr_str)



def img_tran(img):


    path1 = "./Downloads/化学数据/hzj0812_test__/1.6-4_1ver__0_.png"
    path2 = "/media/l/6069-F139/ZZH/0914-Ca__/2-5_1hor__1_.png"
    # img1 = cv2.imread(path1)
    # img2 = cv2.imread(path2)
    # np.savetxt('/media/l/6069-F139/ZZH/R1.txt',img1[:,:,1])
    # np.savetxt('/media/l/6069-F139/ZZH/R2.txt',img2[:,:,1])
    img = cv2.imread(path1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)

    img_flat = img.flatten()
    scatter_3d = go.Scatter3d(x=img_flat[:-2:3],
                              y=img_flat[1:-1:3],
                              z=img_flat[2::3],
                              mode='markers',
                              marker=dict(size=1.5,
                                          color='rgb(93,145,153)'))

    layout = go.Layout(margin=dict(t=10, r=10, b=10, l=10,
                                   pad=50),
                       scene=dict(xaxis_title='Red',
                                  yaxis_title='Green',
                                  zaxis_title='Blue'),
                       autosize=False,
                       width=800,
                       height=800)

    fig = go.Figure(data=scatter_3d, layout=layout)

    plot(fig)

cunt('./Downloads/化学数据/ZZH/0914-Ca__')
print("")
