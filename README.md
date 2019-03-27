# Simultaneous Localization and Mapping
----
Apply SLAM on maps to accuraize the robot trajectory and texure the map image with real time color.

## Requirements
----
 - matplotlib>=2.2
 - numpy>=1.14

## Usage
----
#### Load Data
```sh
load_data.py
```
Load all the data necessary for running the code.
#### Get Parameters
```sh
utils.py
```
Before processing the data set, we need to calculate the necessary parameters to run SLAM. This file includes all the function used in this project, including calculating encoder, imu, lidar and the color mapping image time stamps.
#### Motion Model and Dead-Reckoning
```sh
MM&DR.py
```
This file shows all the calculation to perform the motion model trajectory and dead-reckoning after after using the bersenham function. It prints out two graphs.
#### Particle Filter and Texture Mapping
```sh
PR&TM.py
```
We will apply a particle filter onto the exsistance mapping function to see if this inhance the accuracy of mapping. Texture mapping is to add color to the mapping area by utilizing real image. This prints out three graphs including the best particle trajectory from particle filter.
