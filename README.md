## Surface Normal from Dense Depth Map 
A static filter using Haar like kernels to estimate surface normal maps from dense depth maps. It is implemented as a feedforward neural network in PyTorch and its  usage can be inferred from [demo.py](https://github.com/Ruthrash/surface_normal_filter/blob/master/demo.py)

# Example 
For a depth image like the one below, 

![alt text](https://github.com/Ruthrash/surface_normal_filter/blob/master/depth_img.jpg) 

the output surface normal map is 

![alt text](https://github.com/Ruthrash/surface_normal_filter/blob/master/surface_img.jpg)

Individual X,Y,Z components of the surface normals can be seen as,

![alt text](https://github.com/Ruthrash/surface_normal_filter/blob/master/surface_img_component_x.jpg)
![alt text](https://github.com/Ruthrash/surface_normal_filter/blob/master/surface_img_component_y.jpg)
![alt text](https://github.com/Ruthrash/surface_normal_filter/blob/master/surface_img_component_z.jpg)
