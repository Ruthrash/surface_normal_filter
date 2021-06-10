from os import stat
from os.path import supports_unicode_filenames
from numpy.lib.npyio import mafromtxt
import torch
from torch.functional import norm
#from torch._C import double
import torch.nn as nn
import torch.nn.functional as F

import json
import numpy as np
import timeit    
import cv2

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.convDelYDelZ = nn.Conv2d(1, 1, 3)
        self.convDelXDelZ = nn.Conv2d(1, 1, 3)
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu" 
        self.device = torch.device(dev)
        print("dev!!!", dev)  

    def forward(self, x):
        start = timeit.default_timer()
        #x = x.to(self.device)
        nb_channels = x.shape[0]
        h, w = x.shape[-2:]


        delzdelxkernel = torch.tensor([[0.00000, 0.00000, 0.00000],
                                        [-1.00000, 0.00000, 1.00000],
                                        [0.00000, 0.00000, 0.00000]])
        delzdelxkernel = delzdelxkernel.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)#.to(self.device)
        delzdelx = F.conv2d(x, delzdelxkernel)

        delzdelykernel = torch.tensor([[0.00000, -1.00000, 0.00000],
                                        [0.00000, 0.00000, 0.00000],
                                        [0.0000, 1.00000, 0.00000]])
        delzdelykernel = delzdelykernel.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)#.to(self.device)

        delzdely = F.conv2d(x, delzdelykernel)

        delzdelz = torch.ones(delzdely.shape, dtype=torch.float64)#.to(self.device)

        surface_norm = torch.stack((-delzdelx,-delzdely, delzdelz),2)
 
        surface_norm = torch.div(surface_norm,  norm(surface_norm, dim=2))
        surface_norm_viz = torch.mul(torch.add(surface_norm, 1.00000),127 )
        end = timeit.default_timer()
        print("torch method time", end-start)
        return surface_norm_viz

def main():
    net = Net()
    net.to(net.device)
    with open("depth_images.json",'r') as file_:
        data_dict = json.load(file_)
    time_idx = 15; idx = 1 

    
    img = np.array(data_dict[str(time_idx)][idx], dtype=np.float32)
    start = timeit.default_timer()
    normal_img = np.zeros((129,129,1),dtype=np.float32)
    for i in range(img.shape[0]-1):
        for j in range(img.shape[0]-1):
            if i ==0 or j==0:
                normal_img[i,j,0] = 0
                #normal_img[i,j,1] = 0
                #normal_img[i,j,2] = 0
            else:
                dzdx = img[i+1,j] - img[i-1,j]
                dzdy = img[i,j+1] - img[i,j-1]
                normal = np.zeros((3,1),dtype=np.float32)
                normal[0] = -dzdx
                normal[1] = -dzdy
                normal[2] = 1.0000

                norm = np.linalg.norm(normal)
                normal = normal/norm
                #normal = normal + 1.0
                #if(normal[2] > 0):
                #    print("no")
                normal_img[i,j,0] = int((normal[2]+1.00000)*127.5) #normal[0]#
                #normal_img[i,j,1] = int((normal[1]+1.00000)*127.5) #normal[1]#
                #normal_img[i,j,2] = int((normal[2]+1.00000)*127.5) #normal[2]#

    end = timeit.default_timer()     
    print("1",normal_img[1:-1,1:-1,0])
    #print('2',normal_img[1:-1,1:-1,1])
    #print('3',normal_img[1:-1,1:-1,2])
    #print("time", end-start)
    cv2.imwrite("surface_img_"+str(time_idx)+"__"+str(idx)+".jpg", normal_img)   
    
    
    
    
    
    
    
    
    
    
    
    start = timeit.default_timer()
    img = torch.from_numpy(img)
    img = img[None,None,:,:]
    print('image',img.shape, img.is_cuda)

    normal_output = net.forward(img).cpu().numpy()[0,0,:,:,:]
    print('surface', normal_output.shape )
    
    normal_img = np.zeros((126,126,1),dtype=float)
    normal_img[:,:,0] = normal_output[2,:,:]  
    #normal_img[:,:,1] = normal_output[1,:,:]
    #normal_img[:,:,2] = normal_output[2,:,:]
    print('normal',normal_output.shape, normal_output.is_cuda)

    cv2.imwrite("torch_surface_img_"+str(time_idx)+"__"+str(idx)+".jpg", normal_img)    
    #print(normal_img.channels())
    #normal_img = normal_img
    #

    
        
    #grayImage = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("gray_surface_img_"+str(time_idx)+"__"+str(idx)+".jpg", grayImage)
if __name__ == "__main__":
    pass
    main()