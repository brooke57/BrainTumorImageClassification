# Brain Tumor Image Classification

## Overview
Brain Tumors represent some of the most deadliest forms of cancer. Unfortunately, the five year survival rate for those diagnosed with any kind of brain tumor is [36%](https://www.cancer.net/cancer-types/brain-tumor/statistics), while the survival rate for more serious brain tumor types, such as a grade four glioma is [6-22%](https://moffitt.org/cancers/brain-cancer/survival-rate/). In addition, even for brain tumors that are less lethal and benign, having any kind of brain tumor is dangerous because it can put [extra pressure on the brain](https://www.hopkinsmedicine.org/health/conditions-and-diseases/brain-tumor) or block the flow of cerebrospinal fluid in the brain, which can lead to some serious health problems. This is why early and accurate detection and classification of brain tumors is vital. Brain cancer survival rates could be increased if tumors could be identified earlier and more accurately, which AI methods have the potential to do.

Brain tumors can be difficult to diagnose from an MRI image, and [research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8508169/) has shown that machine learning models are better at detecting and diagnosing brain tumors than humans. This is because computers are able to see MRI images as ["three dimensional datasets"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8508169/) and analyze them down to the three-dimensional, single-unit voxels. This gives computers the ability to find subtle patterns not visible to the naked eye.

## Business Problem
The organization *Doctors without Borders* is constantly seeking to improve healthcare in developing nations. Artificial Intelligence could greatly assist them in these efforts, by assisting Doctors in the diagnosis of brain tumors. For the diagnosis of a brain tumor, a neurosurgeon is required to make the diagnoses from looking at the MRI, and in third world countries seasoned neurosurgeons are somewhat rare. A machine learning tool which could distinguish between normal and tumorous brain MRIs, thereby flagging those with tumors for further analysis and classification of tumor type by a qualified doctor, would be of great value to *Doctors without Borders* as they seek to improve healthcare in developing countries. In a developing nation with fewer seasoned neurosurgeons, where other types of doctors have to step in who may still be learning to detect and diagnose tumors from an MRI, this would be especially valuable. Having doctors use this machine learning model as a supplemental tool could help speed up tumor detection and accuracy. Additionally, using this model could potentially cut down on the physician time required to analyze patient scans. 

## Data
For this project, I used a kaggle dataset (["Brain Tumor Classification (MRI)"](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)) which is divided into four different tumor types: Glioma, Meningioma, Pituitary, and no tumor. Since I used this data for both a binary and multiclass classification project, I used the data in slightly different ways, depending on the project type. For the binary classification project, I downloaded the kaggle dataset and combined all the tumor scans into one category and then uploaded it onto kaggle, to create a binary classification data set. This resorted dataset can be found [here](https://www.kaggle.com/brookejudithsmyth/resortedbraintumorclassificationmridata). For the multiclass project, I simply used the data as it exists in the original kaggle dataset. 
There are a variety of different planes, or perspectives, from which the scans are taken; sagital, coronal, and transverse. The MRIs are also taken with a variety of different methods, namely T1, T2, and FLAIR. Each of these different methods results in an MRI image with varying levels of brightness and contrast. The data used to train the model consists of 2,475 tumor MRIs and 395 normal MRIs of different sizes and is imbalanced. Each image uses all three color channels.


## Preprocessing
In order to maximize GPU time for running Neural Networks (since convolutional neural networks require lots of computational power), I used both Google Colab and Kaggle.

I chose to preprocess the images using the ImageDataGenerator from Keras, which has several advantages. It facillitates smooth resizing and rescaling of images, as well as easy creation of a validation dataset. It also has great data augmentation capabilities; it can generate new images based off of the input in real time during each epoch, in order to mimic the effect of having a greater number of images.


## Modeling Results

**Binary Classification Results**\
Throughout this convolutional neural network modeling process, many different iterations were run. In the end, the iteration called "Incorporating Class Weights into Pretrained VGG-19 (Final Model)" yielded the best results. This model iteration has a base that is a pretrained VGG-19 network, with a flatten layer and two dense layers on top, and all of the VGG-19 layers frozen. It accounts for class weights, giving the minority class of "no tumor" images a weight of three.  This model iteration had a validation accuracy of 95%, a loss of 12%, recall of 100%, and a precision of 95%. The resulting confusion matrix is shown below, where it is clear that true positives and true negatives are being maximized. 

![Screen Shot 2022-02-08 at 10 11 00 PM](https://user-images.githubusercontent.com/68525050/153120742-29ad7a43-c746-40d5-9a89-738c2fba74d2.png)

**Multiclass Classification Results**
I once again used the pretrained VGG-19 Network. The iteration which produced the best results yielded an accuracy of 78%, recall of 73%, precision of 85%, and an F1 score of 78%. The model was better at correctly identifying some tumor types than others; it correctly identified 83% of no tumor scans, 90% of pituitary tumors, 55% of Meningiomas, and 69% of Gliomas.

![Screen Shot 2022-02-08 at 11 26 09 PM](https://user-images.githubusercontent.com/68525050/153128604-aada37d5-7fda-4147-94f1-8b2215b46817.png)

## Model Interpretability

One of the drawbacks to using neural networks is that it is a "black box classifier," meaning that it is hard to determine what the model is picking up on in order to classify the data. However, there is a python library called LIME that lets you explore what the model is picking up on to help it decide how to classify. When I used LIME on images classified with the binary classification model, the most valuable insight was the fact that the eye region (which sometimes shows up in MRIs of the brain) was causing the model to think there was a tumor in the image, due to the fact that eyes in MRIs look very similar to cystic (fluid filled) gliomas. The image below shows an example of an MRI with no tumor that was classified as having a tumor, most likely due to the presence of the eye region in the MRI. A further analysis can be found [here.](https://github.com/brooke57/BrainTumorImageClassification/blob/main/Exploring_the_Black_Box_with_Lime.ipynb)
![Screen Shot 2022-02-08 at 11 51 41 PM](https://user-images.githubusercontent.com/68525050/153130188-501ebe84-f985-4050-8587-ebcb9b02547a.png)

# Conclusions
With high accuracy, recall, and precision, this neural network model would serve as a competent supplementary tool for physicians, physician assistants, and nurses whose specialty may not be may be in brain tumor detection. It would help them more quickly and accurately detect brain tumors and flag scans which require further analysis by neurosurgeons, potentially giving them more time and energy to focus on other patients. These results together have the potential to improve health outcomes for patients in developing nations. 

## Further Steps
One of the most important future steps to be taken is deployment of this neural network model, perhaps in the form of a web app accessible to the relevant mdeical personell. Also, the multiclass classification neural network could use further tuning, to increase its accuracy. In addition, it would be beneficial to focus on the different tumor grades that exist among each tumor types, such as further classifying gliomas (the most harmful form of brain tumor) based on their grade, which determines how dangerous the tumor is.


## Information

- For information regarding the resorting of images from the kaggle dataset to be a binary classification problem, please see [this notebook](https://github.com/brooke57/BrainTumorImageClassification/blob/main/Supplemental_Notebooks/Renaming_Tumor_Images.ipynb), and please see the top of the [final notebook](https://github.com/brooke57/BrainTumorImageClassification/blob/main/Final_Binary_Brain_Tumor_Classification.ipynb) for details on how to load the data onto Google Colab. For information on google colab environment, please see [this file](https://github.com/brooke57/BrainTumorImageClassification/blob/main/Supplemental_Notebooks/Environment_Requirements.rtf).
- To see all models that were run, in addition to the ones in the final notebook, go to the ["All_Models_for_Binary_Brain_Tumor_Classification.ipynb](https://github.com/brooke57/BrainTumorImageClassification/blob/main/Supplemental_Notebooks/All_Models_for_Binary_Brain_Tumor_Classification.ipynb) notebook.
- For information on training the multiclass classification model, go to [this notebook](https://github.com/brooke57/BrainTumorImageClassification/blob/main/Supplemental_Notebooks/transfer-learning-multiclass-brain-tumor.ipynb)
