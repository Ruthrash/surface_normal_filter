import json 
from surface_normal_filter import SurfaceNet
import torch 
import numpy as np 
import cv2 


filter_ = SurfaceNet()

with open('data/depth_images_sample.json','r') as file_:
	images_dict = json.load(file_) 

img = np.array(images_dict['images'][0], dtype=np.float32)


depth_img = img/(np.amax(img))
depth_img = depth_img*255.0
cv2.imwrite("data/depth_img.jpg", depth_img)


img = torch.from_numpy(img)
img = img[None,None,:,:]




surface_norm = filter_(img)

# * normalizing vector space from [-1.00,1.00] to [0,255] for visualization processes
surface_norm_viz = torch.mul(torch.add(surface_norm, 1.00000),127 )

normal_output = surface_norm_viz.cpu().numpy()[0,0,:,:,:]
normal_img = np.zeros((126,126,3),dtype=float)
normal_img[:,:,0] = normal_output[0,:,:]  
normal_img[:,:,1] = normal_output[1,:,:]
normal_img[:,:,2] = normal_output[2,:,:]
cv2.imwrite("data/surface_img.jpg", normal_img)  

component_img = np.zeros((126,126,1),dtype=float)
component_img = normal_output[0,:,:] 
cv2.imwrite("data/surface_img_component_x.jpg", component_img)  

component_img = normal_output[1,:,:] 
cv2.imwrite("data/surface_img_component_y.jpg", component_img)  

component_img = normal_output[2,:,:] 
cv2.imwrite("data/surface_img_component_z.jpg", component_img)  


