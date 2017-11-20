# Follow Me! Deep Learning Project
# Robotics Nanodegree
# Udacity

## Content
[1. Project Objective](#1.-project-objective)  
[2. Data Collection Process](#2.-data-collection-process)  
[3. Network Architecture](#3.-network-architecture)  
[4. Training](#4.-training)  
[5. Results and Improvements](#5.-results-and-improvements)  

[test-scenario1]: ./writeup_images/test-scenario1.png
[test-scenario2a]: ./writeup_images/test-scenario2a.png
[test-scenario2b1]: ./writeup_images/test-scenario2b1.png
[test-scenario2b2]: ./writeup_images/test-scenario2b2.png
[test-scenario31]: ./writeup_images/test-scenario31.png
[test-scenario32]: ./writeup_images/test-scenario32.png

## 1. Project Objective 
The objective of this project is to train a fully convolutional network with images from various scenarios regarding a virtual environment where  a hero is spawned along with people simulations and the corresponding surroundings. The purpose is that the drone using the trained model is able to follow the hero regardless of the obstacles and people along its path. To carry out this, each pixel in the image is classified as either belonging to the target hero or as part of something or someone else in the environment.

The results for the trained neural network will be considered satisfactory in the case that the generated model (file with extension .h5) obtains an accuracy greater or equal to 0.40 in the intersection over union metric (IoU).
Along this writeup, I will be explaining the remaining points indicated in the content list: data collection process, network architecture, training process and last the results and possible areas of improvement.

## 2. Data Collection Process
For carrying out the data collection process, I used the QuadRotor simulator supplied by Udacity. Here, the objective was to take several collections of images involving the target individual (the hero) in several scenarios. In my case I recorded 3 different scenarios indicated below:

**Scenario 1:**  
In this scenario I defined patrol points around an area going from a high height distance down to an earth high distance from the hero with crossing paths (view image below). In this case I didn't spawn any people in the scenario, instead the scenario already contains additional obstacles such as trees in the middle.

![Test scenario 1][test-scenario1]

**Scenario 2:**  
In this scenario, I followed the recommendations in the project guidelines to spawn the hero in a zigzag trajectory and at the same time move the drone along in a linear trajectory. At first I tried the following combination:

![Test scenario 2 case a][test-scenario2a]

But after beginning recording I realized I had defined too many patrol points and that the drone was recording too much content without the hero nor any relevant data. For this reason I simplified both trajectories into a more simple path so that there would be many more crossings between the drone and the hero in several angles and positions:

![Test scenario 2 case b image 1][test-scenario2b1]
![Test scenario 2 case b image 2][test-scenario2b2]

This path happened to be much more efficient than the previous one, resulting in a larger number of images containing the hero.

**Scenario 3:**  
In this scenario I drew a complex path inside a specific circular area for the hero and spawned a considerable amount of people from the middle of the area, having the drone patrol all over this same area, in order to obtain images of the hero in between varying amount of people:

![Test scenario 3 image 1][test-scenario31]
![Test scenario 3 image 2][test-scenario32]

The data recorded for each of the scenarios is the following:

Run number | Data sets collected | Images per data set
------------ | ------------- | -------------
Run 1 | 133 data set | 4 images per data set
Run 2 | 584 data set | 4 images per data set
Run 3 | 1054 data set | 4 images per data set
**Total** | **1771 data set** | **4 images per data set**


## 3. Network Architecture  

The fully convolutional network I used for this project consists of an encoding stage (2 encoding layers) connected to a decoding stage (2 decoding layers) via a 1x1 convolutional layer (view table below).

We know that the original image size of 256 x 256 x 3 has been resized to 160 x 160 x 3 with the data_iterator.py as it is indicated in the jupyter notebook for the project.

By using a 1 x 1 convolutional layer we can take in any image size, as opposed to a fully connected layer which would then require a very specific set of input dimensions.

*Encoders*  
The role of the encoders in the network is to identify the important features in the images being loaded, then keep those features in memory and remove any added noise, decreasing the width and height at the same time they increase the depth of the layer.

For the first encoding layer I chose the following parameters:



After carrying out some small tests and checking with other students in the slack forum I decided to go on with the following cofiguration since it seemed to be the one giving better results at the same time it kept the solution simple and avoid computational excessive time:

Layer name | Layer description | Dimensions
------------ | ------------- | -------------
**Input** | Input data | 160 x 160 x 3
Layer 1 | Encoder | 80 x 80 x 32
Layer 2 | Encoder | 40 x 40 x 64
Layer 3 | Convolution | 40 x 40 x 128
Layer 4 | Decoder | 40 x 40 x 64
Layer 5 | Decoder | 80 x 80 x 32
**Output** | Output data | 160 x 160 x 3

This configuration part was actually quite interesting in order to have a deeper knowledge on 





## 4. Training  
Since I haven't had the AWS p2x.large instance approved yet, I tried carrying out the training in another instance I already had approved in AWS with unsatisfactory results. For this reason I ended up executing the training process in my own computer (MacBook Pro with NVIDIA GeForce GT 750M graphics card using CUDA and CUDNN).

First I carried out a couple of executions with a reduced number of epochs in order to optimize the amount of time it took for the training process and avoid having the process working for too long. 

10 epochs => approximately 3 hours


I trained the network in my o


## 5. Results and Improvements  