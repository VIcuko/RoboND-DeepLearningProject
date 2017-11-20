# Follow Me! Deep Learning Project
# Robotics Nanodegree
# Udacity

## Contents
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

Scenario 1:
In this scenario I defined patrol points around an area going from a high height distance down to an earth high distance from the hero with crossing paths (view image below). In this case I didn't spawn any people in the scenario, instead the scenario already contains additional obstacles such as trees in the middle.

![Test scenario 1][test-scenario1]

Scenario 2:
In this scenario, I followed the recommendations in the project guidelines to spawn the hero in a zigzag trajectory and at the same time move the drone along in a linear trajectory. At first I tried the following combination:

![Test scenario 2 case a][test-scenario2a]

But after beginning recording I realized I had defined too many patrol points and that the drone was recording too much content without the hero nor any relevant data. For this reason I simplified both trajectories into a more simple path so that there would be many more crossings between the drone and the hero in several angles and positions:

![Test scenario 2 case b image 1][test-scenario2b1]
![Test scenario 2 case b image 2][test-scenario2b2]

This path happened to be much more efficient than the previous one, resulting in a larger number of images containing the hero.

Scenario 3:
In this scenario I drew a complex path inside a specific circular area for the hero and spawned a considerable amount of people from the middle of the area, having the drone patrol all over this same area, in order to obtain images of the hero in between varying amount of people:

![Test scenario 3 image 1][test-scenario31]
![Test scenario 3 image 2][test-scenario32]

The data recorded for each of the scenarios is the following:

Run number | Data sets collected | Images per data set
------------ | -------------
Run 1 | 133 data set | 4 images per data set
Run 2 | 584 data set | 4 images per data set
Run 3 | 1054 data set | 4 images per data set
**Total** | 1771 data set | 4 images per data set


## 3. Network Architecture  



## 4. Training  


## 5. Results and Improvements  