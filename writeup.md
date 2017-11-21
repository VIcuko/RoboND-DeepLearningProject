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
[network-diagram]: ./writeup_images/network-diagram.png
[training-curve]: ./writeup_images/training-curve.png
[training-result]: ./writeup_images/training-result.png

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

After carrying out some small tests and checking with other students I decided to go on with the following cofiguration since it seemed to be the one giving better results at the same time it kept the solution simple and avoid computational excessive time:

Layer name | Layer description | Dimensions
------------ | ------------- | -------------
**Input** | Input data | 160 x 160 x 3
Layer 1 | Encoder | 80 x 80 x 32
Layer 2 | Encoder | 40 x 40 x 64
Layer 3 | Convolution | 40 x 40 x 128
Layer 4 | Decoder | 40 x 40 x 64
Layer 5 | Decoder | 80 x 80 x 32
**Output** | Output data | 160 x 160 x 3

![Network diagram][network-diagram]

Below I'm describing the steps for this configuration in further detail:

*Encoders*  
The role of the encoders in the network is to identify the important features in the images being loaded, then keep those features in memory and remove any added noise, decreasing the width and height at the same time they increase the depth of the layer.

For the encoder block, I included a separable convolution layer using the separable_conv2d_batchnorm() function as follows:

```python
def encoder_block(input_layer, filters, strides):
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```

For the first encoding layer I chose the following parameters:

Encoder layer | Filters | Strides
------------ | ------------- | -------------
Layer 1 | 32 | 2
Layer 2 | 64 | 2

The choice of using 2 strides is to lose some height and width for the following layer enabling a better processing time for the training process.

In order to determine the size of the following layers, I have used the equation shown during the course:  

`(N-F + 2*P)/S + 1`  

Substituting the corresponding values in the equation we get the following:  

`layer1_height/width = (160-3 + 2*1)/2 +1 => 80`

Therefore the layer 1 dimensions will be: 80 x 80 x 32 (since I chose 32 filters) 

Applying the same logic, layer 2 will have a dimensions of: 40 x 40 x 64 (since I chose 64 filters)

The resulting code will be the following inside the fcn_model function:

```python
layer1 = encoder_block(inputs, filters=32, strides=2)
layer2 = encoder_block(layer1, filters=64, strides=2)
```

*Convolution layer*  
The convolution layer is placed between the encoder layers and the decoder layers.
I have used a 1 x 1 convolution layer with the purpose of semantic segmentation. This has several advantages for the network:

1. As indicated before, it helps make the network more flexible by allowing different size input images.
1. It preserves the spatial information of the image at the same time it decreases the dimension.

In this case I used 128 filters for this layer.

Therefore, the corresponding line of code within the fcn_model function is:

```python
layer3 = conv2d_batchnorm(layer2, filters=128, kernel_size=1, strides=1)
```

*Decoder*
Within the decoding process, the upsampling part is very important, in order to transform the downsampled image back into the resolution of the original input image. 

In order to carry out the upsample process I used the bilinear upsample function provided in the notebook:   

```python
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```

Additionally, I have concatenated the layers in order to help skip connections and added some separable convolution layers.

The resulting code for the decoding block is the following:

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    upsampled_layer = bilinear_upsample(small_ip_layer)
    concatenated_layer = layers.concatenate([upsampled_layer, large_ip_layer])
    output_layer = separable_conv2d_batchnorm(concatenated_layer, filters)
    output_layer = separable_conv2d_batchnorm(output_layer, filters)
    return output_layer
```

Regarding the fcn_model function, the selection for the decoders was the one required to scale back up the image from the convolutional layer in order to ouput it. Having therefore the 1st decoding layer 64 filters and the 2nd layer 32 filters. The resulting lines in the fcn_model function are the following:

```python
layer4 = decoder_block(layer3, layer1, filters=64)
layer5 = decoder_block(layer4, inputs, filters=32)
```

*Output*
Finally I obtained the output by applying the softmax activation function to generate probability predictions for each pixel.

The whole fcn_model function therefore results in the following code:

```python
def fcn_model(inputs, num_classes):
    layer1 = encoder_block(inputs, filters=32, strides=2)
    layer2 = encoder_block(layer1, filters=64, strides=2)
    layer3 = conv2d_batchnorm(layer2, filters=128, kernel_size=1, strides=1)
    layer4 = decoder_block(layer3, layer1, filters=64)
    layer5 = decoder_block(layer4, inputs, filters=32)
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(layer5)

```

Here the resulting layer will be the same width and height as the input image and in this case 3 classes deep.

## 4. Training  
Since I haven't had the AWS p2x.large instance approved yet, I tried carrying out the training in another instance I already had approved in AWS with unsatisfactory results. For this reason I ended up executing the training process in my own computer (MacBook Pro with NVIDIA GeForce GT 750M graphics card using CUDA and CUDNN) until I was able to execute it in the p2x.large instance in AWS.

First I carried out a couple of executions with a reduced number of epochs in order to optimize the amount of time it took for the training process and avoid having the process working for too long, and then I progressively modified the parameters increasingly and therefore increasing the amount of time required to process the training data.

The definition of each component of the hyperparameters is the following, being the different values used for each case defined in the [Results and Improvements](#5.-results-and-improvements) section below.

*batch_size*: number of training samples/images that get propagated through the network in a single pass.

*num_epochs*: number of times the entire training dataset gets propagated through the network.

*steps_per_epoch*: number of batches of training images that go through the network in 1 epoch. One recommended value to try would be based on the total number of images in training dataset divided by the batch_size.

*validation_steps*: number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset.

*workers*: maximum number of processes to spin up.

workers: This is the number of processes we can start on the CPU/GPU.

## 5. Results and Improvements

My initial hyperparameters were the following:

```python
learning_rate = 0.010
batch_size = 32
num_epochs = 10
steps_per_epoch = 100
validation_steps = 50
workers = 4
```

This process took approximately 2 hours to execute (10-15 mins per epoch) and gave a final score of 0.32, which isn't completely bad, but could definitely be improved.

Afterwards, after several minor tests I decided to increase the steps_per_epoch to 200 and the number of epochs from 10 to 50. The hyperparameters for this case were the following:

```python
learning_rate = 0.010
batch_size = 32
num_epochs = 50
steps_per_epoch = 200
validation_steps = 50
workers = 4
```

This process took quite a while (approximately 16 hours) and in the end it obtained a final score of 0.35. This was definitely an improvement (3 % more), although I wasn't completely convinced by the relation between result and improvement, since it took quite a while to process this case.

After this I had my AWS instance approved and I began to try out various parameters and couldn't find an effective way to improve the results, so I recurred to increasing the training data by carrying out additional image gathering.

After this process, I trained again the network with the new added content and was quite surprised about how this had improved the result. In order to test this new data set I began with the following hyperparameters:

```python
learning_rate = 0.010
batch_size = 32
num_epochs = 10
steps_per_epoch = 200
validation_steps = 50
workers = 40
```
(In this case I increased the number of workers given I was now executing the code in AWS instead of my own computer)

After training the network again with these parameters, I obtained a final score of 0.398, being quite close to the required result, I just needed to improve it a bit more.

Therefore, I slightly increased the number of epochs. First I increased it to 20, but the result decreased considerably to 0.35. I then tried using epochs closer to 10 like 11 - 12, but the result was still worse than the previous case.

For this reason I ended up recurring to modify the encoder, convolutional and decoder layers, applying filters of 64 x 128 x 256 instead of the previous 32 x 64 x128, modifying therefore the fcn_model function to the following, even though this basically doubled the execution time required for the training process, but since I was already executing it in AWS, it was still acceptable.

```python
def fcn_model(inputs, num_classes):
    number of filters) increases.
    layer1 = encoder_block(inputs, filters=64, strides=2)
    layer2 = encoder_block(layer1, filters=128, strides=2)
    layer3 = conv2d_batchnorm(layer2, filters=256, kernel_size=1, strides=1)
    layer4 = decoder_block(layer3, layer1, filters=128)
    layer5 = decoder_block(layer4, inputs, filters=64)
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(layer5)
```

With this configuration and the following hyperparameters:

```python
learning_rate = 0.010
batch_size = 32
num_epochs = 10
steps_per_epoch = 200
validation_steps = 50
workers = 40
```

I managed to obtain a final score of 0.417 which is at last over the required score of 0.40.

The training curve obtained for this case was the following:

![Training curve][training-curve]

And the resulting images for run_1 seemed to be quite defined and precise:

![Training result][training-result]

*Possible improvements:*

After having carried out many tests and tried out many parameters combinations and realizing an increase in image data would have such a significant impact on the result, I think the following could be improved:

1. As indicated, increasing the training data has a significant impact on the result of the network, although it would have its counterpart, meaning that it would require more time and resources to train the network with all the new data included.

1. I could also increase the depth of the network by adding more convolutions with skipped connections.

1. Perhaps including additional targets in order to differentiate even more the content of the image.



