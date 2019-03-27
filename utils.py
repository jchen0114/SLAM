
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
#%%
def headtail(encoder_stamps, imu_stamps):
    count, countt = 0,0
    for i in encoder_stamps:
        if i < min(imu_stamps):
            count += 1
        elif i > max(imu_stamps):
            countt +=1
    head, tail = count+1, len(encoder_stamps)-countt+1
    return head, tail

def ImuMatchEncoder(imu_av,imu_stamps,encoder_t):
    imu_yaw = imu_av[2] 
    imu_time = imu_stamps
    imu_time_scaled, index_list = [], []

    for i in range(len(encoder_t)):
        diff = list(abs(imu_time - encoder_t[i]))
        diff_min = np.min(diff)
        index = diff.index(diff_min)
        index_list.append(index)
    imu_time_scaled = imu_time[index_list]
    imu_yaw = imu_yaw[index_list]
    
    return imu_time_scaled, imu_yaw

def EncodernImuTime(imu_time_scaled,estamps,head, tail):
    imu_time =[0]
    for i in range(len(imu_time_scaled)):
        if len(imu_time_scaled) > i+1:
            t = imu_time_scaled[i+1]-imu_time_scaled[i]
            imu_time.append(t)
    
    encoder_time = []
    for i in range(len(estamps[head:tail])):
        if len(estamps) > i+1:
            tt = estamps[i+1] - estamps[i]
            encoder_time.append(tt)
    return imu_time, encoder_time


def EncoderV(encoder_counts,encoder_time,head, tail):
    encoder_left = ((encoder_counts[1]+encoder_counts[3])/2 *0.0022)[head:tail]
    encoder_right = ((encoder_counts[0]+encoder_counts[2])/2 *0.0022)[head:tail]
    encoder_avg = ((encoder_left + encoder_right)/2)/encoder_time
    return encoder_avg

def XYTheta(imu_yaw, encoder_time, encoder_avg,head,tail):
    x,y,theta = 0,0,0
    x_pos, y_pos, theta_temp =[],[],[]
    for i in range(tail-head):
        xy = imu_yaw[i]*(encoder_time[i])/2
        x = x+encoder_avg[i]*encoder_time[i]*(np.sin(xy)/xy)*np.cos(theta+imu_yaw[i]*(imu_time[i])/2)
        y = y+encoder_avg[i]*encoder_time[i]*(np.sin(xy)/xy)*np.sin(theta+imu_yaw[i]*(imu_time[i])/2)
        x_pos.append(x)
        y_pos.append(y)
        theta = theta + imu_yaw[i]*(imu_time[i])
        theta_temp.append(theta) 
    
    position = np.stack((x_pos,y_pos))
    position = np.reshape(position,(-1,2))
    return x_pos,y_pos,theta_temp,position



def motion_diff(encoder_avg,imy_yaw,theta_temp,t,encoder_time):      
    
    sinc = np.sin(imy_yaw[t]*encoder_time[t]/2)/(imy_yaw[t]*encoder_time[t]/2)
    dx = encoder_avg[t]*sinc*np.cos(theta_temp[t]+imu_yaw[t]*encoder_time[t]/2)*encoder_time[t]
    dy = encoder_avg[t]*sinc*np.sin(theta_temp[t]+imu_yaw[t]*encoder_time[t]/2)*encoder_time[t]
    dtheta = imu_yaw[t]*encoder_time[t]
    return dx,dy,dtheta

def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()      

def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))


def mapCorrelation(im, x_im, y_im, vp, xs, ys):
  '''
  INPUT 
  im              the map 
  x_im,y_im       physical x,y positions of the grid map cells
  vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
  xs,ys           physical x,y,positions you want to evaluate "correlation" 

  OUTPUT 
  c               sum of the cell values of all the positions hit by range sensor
  '''
  nx = im.shape[0]
  ny = im.shape[1]
  
  xmin = x_im[0]
  xmax = x_im[-1]
  xresolution = (xmax-xmin)/(nx-1)
  ymin = y_im[0]
  ymax = y_im[-1]
  yresolution = (ymax-ymin)/(ny-1)
  
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
			                        np.logical_and((ix >=0), (ix < nx)))
      
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
      
  return cpr

def WorldtoMap(x,y):
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -20  #meters
    MAP['ymin']  = -20
    MAP['xmax']  =  20
    MAP['ymax']  =  20 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
  
    xis = np.ceil((x - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((y - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    return xis,yis

def BodytoWorld(x,y,theta):
    trans = [[np.cos(theta),-(np.sin(theta)),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]
    result = np.dot(trans,np.transpose([x,y,0]))
    return result

def AngletoBody(i,lidar_ranges):
    new_lidar = lidar_ranges[:,i]
    angle = np.arange(-135,135.25,0.25)*np.pi/180.0
    wrong = np.logical_and((new_lidar < 30),(new_lidar> 0.1))
    new_ranges = new_lidar[wrong]
    angle = angle[wrong]
    xl = new_ranges * np.cos(angle)+0.298
    yl = new_ranges * np.sin(angle)
    return xl, yl

def Mapping(mapp,x_og,y_og,xis,yis):
    log_odds_occupied, log_odds_empty = 4, 1/4
    maxlist = np.stack((xis,yis)).T
    for j in maxlist:
        bre = 0
        ex,ey = j[0], j[1]
        bre = bresenham2D(x_og, y_og, ex, ey).T
        #bree = np.stack((bre))
        #bree.astype(int)
        bree = np.delete(bre,len(bre)-1,0)
        xbre,ybre = bree[:,0].astype(int),bree[:,1].astype(int)
        mapp[xbre,ybre] += np.log(log_odds_empty)
        mapp[ex,ey] += np.log(log_odds_occupied)
    return mapp

def ThreeColorMap(m):
    for j in range(len(m)):
        for k in range(len(m)):
            if m[j][k] < 0:
                m[j][k] = -1
            elif m[j][k] == 0:
                m[j][k] = 0
            elif m[j][k] > 0:
                m[j][k] = 1
    return m

#%%
def pixeltoOptical(pixeluv1, flat_depth):
    K = [[585.05108211, 0, 242.94140713],
         [0, 585.05108211, 315.83800193],
         [0, 0, 1]]
    canonical = flat_depth
    result = canonical*np.dot(inv(K),pixeluv1)
    return result

def CameraRotation(pixelincamera):
    rotation_matrix = [[np.cos(0.36),0,np.sin(0.36)],
                        [0,1,0],
                        [-np.sin(0.36),0,np.cos(0.36)]]
    return np.dot(rotation_matrix, pixelincamera)


def pixelBodytoWorld(opticalinbody, theta):
    Transform_matrix = [[np.cos(theta),-np.sin(theta),0],
                        [np.sin(theta),np.cos(theta),0],
                        [0,0,1]]
    result = np.dot(Transform_matrix, opticalinbody)
    return result

def DisMatchRgb(disp_stamps,rgb_stamps):
    dis_time = disp_stamps[0:len(rgb_stamps)]
    dis_scaled, index_list = [], []

    for i in range(len(rgb_stamps)):
        diff = list(abs(dis_time - rgb_stamps[i]))
        diff_min = np.min(diff)
        index = diff.index(diff_min)
        index_list.append(index)
    dis_scaled = dis_time[index_list]
    #imu_yaw = imu_yaw[index_list]2188 2112
    
    return dis_scaled,index_list