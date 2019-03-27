
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
#%%
head, tail = headtail(encoder_stamps, imu_stamps)
encoder_times = encoder_stamps[head: tail] 

imu_time_scaled, imu_yaw = ImuMatchEncoder(imu_angular_velocity,imu_stamps,encoder_times)
imu_time, encoder_time = EncodernImuTime(imu_time_scaled,encoder_stamps,head,tail)
encoder_avg = EncoderV(encoder_counts,encoder_time,head, tail)
x_pos,y_pos,theta_temp,position = XYTheta(imu_yaw, encoder_time, encoder_avg, head, tail)


plt.scatter(x_pos,y_pos)
plt.imshow()
#%%
mapp = np.zeros((1200,1200))
l = lidar_ranges[:,head:tail]
for i in range(0,len(l[0]),20):
    
    x_org = x_pos[i] + 0.298*np.cos(theta_temp[i])
    y_org = y_pos[i] + 0.298*np.sin(theta_temp[i])
    x_og,y_og = WorldtoMap(x_org,y_org)
    
    xl, yl = AngletoBody(i,l)
    
    xyz = BodytoWorld(xl,yl,theta_temp[i])
    
    x2 = x_pos[i] + xyz[0]
    y2 = y_pos[i] + xyz[1]
    xis, yis = WorldtoMap(x2,y2)
    
    
    mapp = Mapping(mapp,x_og,y_og,xis,yis)
      
m = ThreeColorMap(mapp)
fig2 = plt.figure()
plt.imshow(m,cmap="hot")
