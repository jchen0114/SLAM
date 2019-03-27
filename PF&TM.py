
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from numpy.linalg import inv
#%% randomlize particle
angles = np.arange(-135,135.25,0.25)*np.pi/180.0
l = lidar_ranges[:,head:tail]
valid = np.logical_and((l[:,0] < 30),(l[:,0]> 0.1))
l = l[valid]
angles = angles[valid]
lamda = np.zeros((1200,1200))

particle = 100
weight = np.ones(particle)/particle
fill = np.zeros((particle,3))
res,dif = 0.05, 2
x_im = np.arange(-30,30)
y_im = np.arange(-30,30)
x_dif = np.arange(-dif*res,dif*res+res,res)
y_dif = np.arange(-dif*res,dif*res+res,res)

now = np.zeros((3,))
noise1 = np.array([0.05, 0.05, 0.1*np.pi/180])
noise2 = np.array([0.01, 0.01, 0.03*np.pi/180])
bestx, besty, bestxx, bestyy =[],[],[],[]
# Set up first state 
for i in range(particle):
    noises = np.random.randn(1,3)*noise1  
    fill[i,:] = [x_pos[0],y_pos[0],theta_temp[0]]+noises
noww = []
for i in range(1,1200):#np.size(l,1)):   
    print(i)
    for j in range(particle):
        #noises = np.random.randn(1,3)*noise2
        dx,dy,dtheta = motion_diff(encoder_avg,imu_yaw,theta_temp,i,encoder_time)
        fill[j,:] = [fill[j,0]+dx,fill[j,1]+dy,fill[j,2]+dtheta]
        #+noises

    fill[:,2] %= 2*np.pi
    xs = l[:,i]*np.cos(angles)+0.298
    ys = l[:,i]*np.sin(angles)
    xyz = BodytoWorld(xs,ys,theta_temp[i])
    vp = np.vstack((xyz[0], xyz[1]))
    
    cors = []
    temp = np.zeros_like(lamda)
    temp[lamda>0] = 1
    temp[lamda<0] = -1
    for j in range(particle):    
        particle_cor_x, particle_cor_y = x_dif+fill[j,0], y_dif+fill[j,1]
        cor = mapCorrelation(temp,x_im,y_im,vp,particle_cor_x,particle_cor_y)
        #index = np.argmax(cor)
        #fill[j,0] = fill[j,0]+x_dif[index%np.size(y_dif)]
        #fill[j,1] = fill[j,1]+y_dif[index//np.size(x_dif)]
        cors.append(np.max(cor))
            
    cors = weight * np.array(cors)
    weight = softmax(cors)
    best = np.argmax(weight)   
    now = fill[best,:].copy()
    noww.append(now)
    bestx.append(now[0])
    besty.append(now[1])
    
    best_particle_map_x,best_particle_map_y = WorldtoMap(now[0]+0.298*np.cos(now[2]),now[1]+0.298*np.sin(now[2]))
    bestxx.append(best_particle_map_x)
    bestyy.append(best_particle_map_y)
    
    xis,yis =  WorldtoMap(xyz[0]+x_pos[i],xyz[1]+y_pos[i])

    lamda = Mapping(lamda,best_particle_map_x,best_particle_map_y,xis,yis)
    

plt.scatter(bestx,besty)
plt.imshow()
#%%
lamda = ThreeColorMap(lamda)
fig = plt.figure()
plt.imshow(lamda,cmap='hot')
#%%
noww = np.array(noww)
dis_scaled, index_list = DisMatchRgb(disp_stamps, rgb_stamps)
rgbmap = np.zeros((1201,1201,3))
groundtruth = lamda
for i in range(len(groundtruth)):
    for j in range(len(groundtruth[0])):
        if groundtruth[i][j] == 1:
            rgbmap[i][j] = [225/225,225/225,225/225]
        else:
            rgbmap[i][j] = [0,0,0]
#%%
for i in range(len(dis_scaled)):
    print(i)
    dispar = plt.imread('C:/Users/justi/Desktop/win19/276A/ECE276A_HW2/dataRGBD/Disparity20/disparity20_{}.png'.format(i+1))
    rgbimg = plt.imread('C:/Users/justi/Desktop/win19/276A/ECE276A_HW2/dataRGBD/RGB20/rgb20_{}.png'.format(index_list[i]))
    dd = (-0.00304 * dispar +3.31)
    depth = 1.03/dd
    rgbi = np.rint((dd*(-4.5)*1750.46 + 19276)/585.051 + np.arange(0,rgbimg.shape[0]).reshape([-1,1])*526.37/585.051)
    rgbj = np.rint(np.tile((np.arange(0,rgbimg.shape[1]).reshape([-1,1])*526.37+16662)/585.051,(rgbimg.shape[0],1)))
    flat_depth = depth.flatten()
    flat_rgbi = rgbi.flatten()
    flat_rgbj = rgbj.flatten()
    flat_ones = np.ones((len(flat_rgbi)))
    pixeluv1 = np.vstack((flat_rgbi,flat_rgbj,flat_ones))
    pixelinoptical = pixeltoOptical(10*pixeluv1, flat_depth)
    opticalinbody = CameraRotation(pixelinoptical)+np.array([[0.18],[0.005],[0.36]])
    pixelinWorld = pixelBodytoWorld(opticalinbody, noww[i,2])
    
    temp_index = np.where(pixelinWorld[2,:]<3.4)
    index_i = flat_rgbi[temp_index].astype('int')
    index_j = flat_rgbj[temp_index].astype('int')
    map_x, map_y = WorldtoMap(pixelinWorld[0,temp_index] + noww[i][0], pixelinWorld[1,temp_index] + noww[i][1])
    rgbmap[map_x,map_y,:] = rgbimg[index_i,index_j,:]

plt.figure()
plt.imshow(rgbmap)