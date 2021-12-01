# Brain Tumor Image Classification
Using a Convolutional Neural Network to Distinguish between Brain MRIs which show a tumor and those that do not.

## Overview
Brain tumors in particular are very difficult to diagnose from an MRI image, and artificial intelligence methods of identifying and classifying tumors are oftentimes more accurate than manual identification by a radiologist. That is why the development of neural networks and other AI processes for tumor classification is so valuable and important. The survival rate for patients diagnosed with a brain tumor is around 35%. This survival rate could be increased if tumors could be identified earlier and more accurately, which AI methods could help with.

## Business Problem
The organization *Doctors without Borders* is constantly seeking to improve healthcare in developing nations. Artificial Intelligence could greatly assist them in these efforts, by assisting Doctors in the diagnosis of brain tumors. For the diagnosis of a brain tumor, a neurosurgeon is required to make the diagnoses from looking at the MRI, and in third world countries seasoned neurosurgeons are somewhat rare. A machine learning tool which could distinguish between normal and tumorous brain MRIs could help flag scans which requires further analysis (in order to determine exact tumor type) would be of great value to *Doctors without Borders* as they seek to improve healthcare in developing countries. 


## Data
This data is composed of a series of Brain MRIs consisting of scans which contain a tumor and those that do not. The data actually comes from an existing kaggle dataset (["Brain Tumor Classification (MRI)"](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)), which is further divided into three different tumor types: Glioma, Meningioma, and Pituitary. I downloaded this data and combined all the tumor scans into one category and then uploaded it onto kaggle, so that I could make this a binary classification problem. This resported dataset can be found [here](https://www.kaggle.com/brookejudithsmyth/resortedbraintumorclassificationmridata). There are a variety of different planes, or perspectives, from which the scans are taken; some are sagital scans (plane that shows the side of the brain), some are coronal scans (plane that shows the back of the brain, at varying depths), and some are transverse (plane that shows the top of the brain, at varying depths; like a bird's eye view). The dataset contains MRIs taken with a variety of different methods, namely T1, T2, and FLAIR. Each of these different methods results in an MRI image with varying levels of brightness and contrast. The data contains 2,764 tumor MRIs and 500 normal MRIs, so the dataset is very imbalanced. Each image uses all three color channels, and each is a different size, so I standardized all images to be 200 x 200 x 3.

![Screen Shot 2021-12-01 at 3 29 58 PM](https://user-images.githubusercontent.com/68525050/144320555-36f6254c-4104-4cb2-a399-a543ff9bfc66.png)


## Preprocessing
I chose to work on this dataset using kaggle because that is where I found the original data, and because of the of the kaggle GPU, because I knew that convolutional neural networks require lots of computational power.

I chose to preprocess the images using the ImageDataGenerator from Keras, which has several advantages, such as: 

- Smooth re-sizing and re-scaling process
- Its Data Augmentation capabilities; it can generate new images based off of the input in real time during each epoch, in order to mimic the effect of having a greater number of images. 
- Easy creation of a validation dataset

Because a total of 2,870 files (number of files in the training set) is a pretty small number of images to use for training a neural network, data augmentation is key. The way I implemented these techiques was by using the ImageDataGenerator from Keras. The features I decided to tweak for augmentation were zoom range, rotation range, brightness range, and horizontal flipping. I decided to provided a range of different zoom values and rotation degrees because how much zoom and the angle of how the brain is positioned in an MRI image can vary a little, and sp producing images with varying levels of zoom and rotation is a realistic way to mimic the effect of having more images. I decided to provide a range for brightness level because as mentioned in the 'Data Understanding' paragraph above, the dataset contains a variety of images with different levels of brightness and contrast, and so producing images with different levels of brightness is a realistic way to mimic the effect of having more images. I decided to flip some images along the horizontal axis, which translates to a left right flip, because regions of the brain are very symmetrical along the left/right axis. I did not include vertical flipping as part of data augmentation, because top/bottom parts of the brain are not symmetrical. I also decided not to shear any images, because shearing stretches and distorts regions of an image, and for brain scans it is very important to preserve the correct anatomical structure of the brain, as discussed in this [this reresearch article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6917660/).


## Modeling Results
My modeling process consisted of a series of Convolutional Neural Networks (CNNs) 


## Information
