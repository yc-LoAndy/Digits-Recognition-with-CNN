# Digits Recognition Deep Learning Model with CNN
Making use of MNIST dataset, I have trained a CNN deep learning model with about 99% accuracy in testing dataset, which is able to be generalized to recognize my own handwritte digits.

This model is trained for practicing purpose only, which means that the accuracy performance is not guaranteed for all kinds of inputs. Also, the input handwritten digits image is required to have clean background; otherwise, the image processing program may not be able to locate the digits successfully.

## Pretraining Phase: DAE
There is a pretraining phase before the model recognizing digits is built.
In the pretraining phase, I built a Denoise Autoencoder(DAE) in order to obtain a set of good weights on each layer, which is later used in the training phase of the digits classifier as the initial value of the weights.

The structure of the encoder is:\
**inputs**(1\*28\*28) -> **Conv2d**(16\*24\*24) -> **Maxpool**(16\*12\*12) -> **Tanh** -> Conv2d(32\*8\*8) -> **Maxpool**(32\*4\*4)

The structure of the decoder is:\
**inputs**(32\*4\*4) -> **ConvTranspose2d**(16\*12\*12) -> **Tanh** -> **ConvTranspose2d**(1\*28\*28) -> **Sigmoid**

This pretraining phase is an unsupervised-training practice which is believed to be able to train the classifier more effectively later when the dataset on hand is insufficient.
However in this case, this practice is solely for practicing and experimental purpose as MNIST definitely provides sufficient images for us to train our classifier.
In fact, I found no difference in the resulting accuracy on test dataset between adopting and not adopting the pretrained weights when training the classifier.

The following picture show the similarity between the original digits and the reconstructed ones by the DAE.\
<img src="https://github.com/yc-LoAndy/Digits-Recognition-with-CNN/blob/main/scripts/pic/ae_reconstruct/Original_images_1.jpg" width="325" height="250"/><img src="https://github.com/yc-LoAndy/Digits-Recognition-with-CNN/blob/main/scripts/pic/ae_reconstruct/Restructured_Image_by_AE_1.jpg" width="325" height="250"/>

## Training Phase
After the pretraining process, we now start the training of the actual digits classifier. This classifier shares the same structure of the encoder built in the pretraining phase to inherit the pretrained weights, while the only difference is that there is a fully-connected(FC) layer by the very end of the model structure.\

The structure of the classifier is:\
**inputs**(1\*28\*28) -> **Conv2d**(16\*24\*24) -> **Maxpool**(16\*12\*12) -> **Tanh** -> Conv2d(32\*8\*8) -> **Maxpool**(32\*4\*4) -> **flatten**(1\*1\*512) -> **FC**(1\*10) -> **Softmax**

The resulting classification accuracy is about 99.15%.\
![Accuracy](https://github.com/yc-LoAndy/Digits-Recognition-with-CNN/blob/main/scripts/pic/acc.png)

## Digits Recognition with my own handdwritten digits
Now I want to try out my own handwritten digits to see if my model can also recognize digits written by myself. Here, we divide the testing into 2 parts: _image processing_ and _testing result_.

### Image Processing
The goal of the image processing is to locate the digits in the image and process them into what the digits in MNIST dataset look like. In this project we use **opencv** to find the contours of the digits and fill the contours in white. Then, we crop the digits into size 28\*28 and save them into _./scripts/pic/digits_.
The following image shows the result of the processing.

<img src="https://github.com/yc-LoAndy/Digits-Recognition-with-CNN/blob/main/scripts/pic/digits/original_digits.jpg" width="305" height="250" /><img src="https://github.com/yc-LoAndy/Digits-Recognition-with-CNN/blob/main/scripts/pic/digits/contor.jpg" width="305" height="250" />

A single digit: <img src="https://github.com/yc-LoAndy/Digits-Recognition-with-CNN/blob/main/scripts/pic/digits/digit_10.jpg" />

### Testing
While testing the model with my own handwritten digits, I found something interesting:\
If the classifier is initialized with the pretrained weights obtained from DAE, the accuracy of the digits recognition is only 42.8% (9 out of 21 digits are correct), while the accuracy is 76.1% (16 out of 21 digits are correct) when not adopting the pretrained weight.
<img src="https://github.com/yc-LoAndy/Digits-Recognition-with-CNN/blob/main/scripts/pic/test_pretrained.png" width="270" height="360" />
<img src="https://github.com/yc-LoAndy/Digits-Recognition-with-CNN/blob/main/scripts/pic/test_no_pretrain.png" width="270" height="360" />

The first thing is that the accuracy gap between test data by MNIST and my own handwritten test is expected, and 76.1% might be acceptable. The reason for the gap is clear: there is noticable difference between my handwritten style and that of MNIST, which is Taiwanese and American.
Since the classifier is trained with American style handwritten digits provided by MNIST, it is expected that the classifier cannot perfectly recognize my handwritten digits.

Second, the observation that adopting the pretrained weights results in significant drop in my own handwritten testing is probabily because of the overfitting issue as the model learn to identify American style handwritten digits more intensively and fail to recognize my hadwriting style.
