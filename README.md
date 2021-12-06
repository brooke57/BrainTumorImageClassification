# Brain Tumor Image Classification
Using a Convolutional Neural Network to Distinguish between Brain MRIs which show a tumor and those that do not.

## Overview
Brain Tumors represent some of the most deadliest forms of cancer. Unfortunately, the five year survival rate for those diagnosed with a grade four glioma is [6-22%](https://moffitt.org/cancers/brain-cancer/survival-rate/). In addition, even for brain tumors that are less lethal and benign, having any kind of brain tumor is dangerous because it can put [extra pressure on the brain](https://www.hopkinsmedicine.org/health/conditions-and-diseases/brain-tumor) or block the flow of cerebrospinal fluid in the brain, which can lead to some serious health problems. This is why early and accurate detection and classification of brain tumors is vital. Brain cancer survival rates could be increased if tumors could be identified earlier and more accurately, which AI methods have the potential to do.

Brain tumors can be difficult to diagnose from an MRI image, and [research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8508169/) has shown that machine learning models are better at detecting and diagnosing brain tumors than humans. This is because computers are able to see MRI images as ["three dimensional datasets"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8508169/) and analyze them down to the three-dimensional, single-unit voxels. This gives computers the ability to find subtle patterns not visible to the naked eye.

## Business Problem
The organization *Doctors without Borders* is constantly seeking to improve healthcare in developing nations. Artificial Intelligence could greatly assist them in these efforts, by assisting Doctors in the diagnosis of brain tumors. For the diagnosis of a brain tumor, a neurosurgeon is required to make the diagnoses from looking at the MRI, and in third world countries seasoned neurosurgeons are somewhat rare. A machine learning tool which could distinguish between normal and tumorous brain MRIs, thereby flagging those with tumors for further analysis and classification of tumor type by a physician, would be of great value to *Doctors without Borders* as they seek to improve healthcare in developing countries. In a developing nation with fewer seasoned physicians, who may still be learning to detect and diagnose tumors from an MRI, this would be especially valuable. Additionally, using this model could potentially cut down on the physician time required to analyze patient scans. 

## Data
This data is composed of a series of Brain MRIs consisting of scans which contain a tumor and those that do not. The data actually comes from an existing kaggle dataset (["Brain Tumor Classification (MRI)"](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)), which is further divided into three different tumor types: Glioma, Meningioma, and Pituitary. I downloaded this data and combined all the tumor scans into one category and then uploaded it onto kaggle, so that I could make this a binary classification problem. This resported dataset can be found [here](https://www.kaggle.com/brookejudithsmyth/resortedbraintumorclassificationmridata). There are a variety of different planes, or perspectives, from which the scans are taken; some are sagital scans (plane that shows the side of the brain), some are coronal scans (plane that shows the back of the brain, at varying depths), and some are transverse (plane that shows the top of the brain, at varying depths; like a bird's eye view). The dataset contains MRIs taken with a variety of different methods, namely T1, T2, and FLAIR. Each of these different methods results in an MRI image with varying levels of brightness and contrast. The data used to train the model consists of 2,475 tumor MRIs and 395 normal MRIs, so the dataset is very imbalanced. Each image uses all three color channels, and each is a different size, so I standardized all images to be 200 x 200 x 3.

![Screen Shot 2021-12-01 at 3 29 58 PM](https://user-images.githubusercontent.com/68525050/144320555-36f6254c-4104-4cb2-a399-a543ff9bfc66.png)


## Preprocessing
In order to maximize GPU time for running Neural Networks (since convolutional neural networks require lots of computational power), I ultimately ended up working on Google Colab for the Final Notebook, although I did begin working with this data on kaggle. 

I chose to preprocess the images using the ImageDataGenerator from Keras, which has several advantages. It facillitates smooth resizing and rescaling of images, as well as easy creation of a validation dataset. It also has great data augmentation capabilities; it can generate new images based off of the input in real time during each epoch, in order to mimic the effect of having a greater number of images.

Because a total of 2,870 files (number of files in the training set) is a pretty small number of images to use for training a neural network, data augmentation is key. The way I implemented these techiques was by using the ImageDataGenerator from Keras. The features I decided to tweak for augmentation were zoom range, rotation range, brightness range, and horizontal flipping. I decided to flip some images along the horizontal axis, which translates to a left right flip, because regions of the brain are very symmetrical along the left/right axis. I did not include vertical flipping as part of data augmentation, because top/bottom parts of the brain are not symmetrical. I also decided not to shear any images, because shearing stretches and distorts regions of an image, and for brain scans it is very important to preserve the correct anatomical structure of the brain, as discussed in this [this reresearch article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6917660/).


## Modeling Results
Throughout this convolutional neural network modeling process, many different iterations were run. In the end, the iteration called "Incorporating Class Weights into Pretrained VGG-19 (Final Model)" yielded the best results. This model iteration has a base that is a pretrained VGG-19 network, with a flatten layer and two dense layers on top, and all of the VGG-19 layers frozen. It accounts for class weights, giving the minority class of "no tumor" images a weight of three.  This model iteration had a validation accuracy of 97%, a loss of 7%, recall of 100%, and a precision of 97%. The resulting confusion matrix is shown below, where it is clear that true positives and true negatives are being maximized. 


![Screen Shot 2021-12-04 at 9 56 02 PM](https://user-images.githubusercontent.com/68525050/144756495-13452643-c6b9-4bb2-a19d-ab43fc01e001.png)

# Conclusions
With such high accuracy, recall, and precision, it is safe to say that this neural network model would be very competent at flagging scans which require further analysis by physicians, potentially giving them more time and energy to focus on other patients. Additionally, it would be a good support tool for physicians learning how to detect brain tumors. These results together have the potential to improve health outcomes for patients in developing nations. 

## Further Steps
One of the most important future steps to be taken is deployment of this neural network model, perhaps in the form of a web app accessible to the relevant mdeical personell. Another very valuable step to be taken is to develop a multiclass classification neural network, which would be able to distinguish between the different main brain tumor types (glioma, meningioma, and pituitary).


## Information

- For information regarding the resorting of images from the kaggle dataset to be a binary classification problem, please see [this notebook](https://github.com/brooke57/BrainTumorImageClassification/blob/main/Renaming_Tumor_Images.ipynb), and please see the top of the [final notebook](https://github.com/brooke57/BrainTumorImageClassification/blob/main/Final_Binary_Brain_Tumor_Classification.ipynb) for details on how to load the data onto Google Colab.
