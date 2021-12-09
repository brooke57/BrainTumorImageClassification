# Brain Tumor Image Classification

## Overview
Brain Tumors represent some of the most deadliest forms of cancer. Unfortunately, the five year survival rate for those diagnosed with any kind of brain tumor is [36%](https://www.cancer.net/cancer-types/brain-tumor/statistics), while the survival rate for more serious brain tumor types, such as a grade four glioma is [6-22%](https://moffitt.org/cancers/brain-cancer/survival-rate/). In addition, even for brain tumors that are less lethal and benign, having any kind of brain tumor is dangerous because it can put [extra pressure on the brain](https://www.hopkinsmedicine.org/health/conditions-and-diseases/brain-tumor) or block the flow of cerebrospinal fluid in the brain, which can lead to some serious health problems. This is why early and accurate detection and classification of brain tumors is vital. Brain cancer survival rates could be increased if tumors could be identified earlier and more accurately, which AI methods have the potential to do.

Brain tumors can be difficult to diagnose from an MRI image, and [research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8508169/) has shown that machine learning models are better at detecting and diagnosing brain tumors than humans. This is because computers are able to see MRI images as ["three dimensional datasets"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8508169/) and analyze them down to the three-dimensional, single-unit voxels. This gives computers the ability to find subtle patterns not visible to the naked eye.

## Business Problem
The organization *Doctors without Borders* is constantly seeking to improve healthcare in developing nations. Artificial Intelligence could greatly assist them in these efforts, by assisting Doctors in the diagnosis of brain tumors. For the diagnosis of a brain tumor, a neurosurgeon is required to make the diagnoses from looking at the MRI, and in third world countries seasoned neurosurgeons are somewhat rare. A machine learning tool which could distinguish between normal and tumorous brain MRIs, thereby flagging those with tumors for further analysis and classification of tumor type by a qualified doctor, would be of great value to *Doctors without Borders* as they seek to improve healthcare in developing countries. In a developing nation with fewer seasoned neurosurgeons, where other types of doctors have to step in who may still be learning to detect and diagnose tumors from an MRI, this would be especially valuable. Having doctors use this machine learning model as a supplemental tool could help speed up tumor detection and accuracy. Additionally, using this model could potentially cut down on the physician time required to analyze patient scans. 

## Data
This data is composed of a series of Brain MRIs consisting of scans which contain a tumor and those that do not. The data actually comes from an existing kaggle dataset (["Brain Tumor Classification (MRI)"](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)), which is further divided into three different tumor types: Glioma, Meningioma, and Pituitary. I downloaded this data and combined all the tumor scans into one category and then uploaded it onto kaggle, so that I could make this a binary classification problem. This resorted dataset can be found [here](https://www.kaggle.com/brookejudithsmyth/resortedbraintumorclassificationmridata). There are a variety of different planes, or perspectives, from which the scans are taken; sagital, coronal, and transverse. The MRIs are also taken with a variety of different methods, namely T1, T2, and FLAIR. Each of these different methods results in an MRI image with varying levels of brightness and contrast. The data used to train the model consists of 2,475 tumor MRIs and 395 normal MRIs of different sizes, so the data requires size standardization and is very imbalanced, which is important to be aware during the modeling process. Each image uses all three color channels.

Another thing to be aware of in the modeling process are the two types of error that the neural network model can make; false positives and false negatives. A false positive occurs when the model identifies a normal MRI scan as having a tumor, and a false negative occurs when a MRI scan with a tumor is identified by the model as normal. In this case, a false negative is worse than a false positive, because this means that a petient who has a brain tumor does not get treatment, and could develop worse health problems. It is ideal to minimize both errors, but during the modeling process I focused more on minimizing false negatives.

![Screen Shot 2021-12-08 at 2 11 51 PM](https://user-images.githubusercontent.com/68525050/145277556-9568d454-72af-4b66-bd2a-e8a09abb5785.png)

## Preprocessing
In order to maximize GPU time for running Neural Networks (since convolutional neural networks require lots of computational power), I ultimately ended up working on Google Colab for the Final Notebook, although I did begin working with this data on kaggle. 

I chose to preprocess the images using the ImageDataGenerator from Keras, which has several advantages. It facillitates smooth resizing and rescaling of images, as well as easy creation of a validation dataset. It also has great data augmentation capabilities; it can generate new images based off of the input in real time during each epoch, in order to mimic the effect of having a greater number of images.


## Modeling Results
Throughout this convolutional neural network modeling process, many different iterations were run. In the end, the iteration called "Incorporating Class Weights into Pretrained VGG-19 (Final Model)" yielded the best results. This model iteration has a base that is a pretrained VGG-19 network, with a flatten layer and two dense layers on top, and all of the VGG-19 layers frozen. It accounts for class weights, giving the minority class of "no tumor" images a weight of three.  This model iteration had a validation accuracy of 97%, a loss of 7%, recall of 100%, and a precision of 97%. The resulting confusion matrix is shown below, where it is clear that true positives and true negatives are being maximized. 

![Screen Shot 2021-12-08 at 2 14 30 PM](https://user-images.githubusercontent.com/68525050/145277925-f60aa1d2-00f3-4945-88a6-94ff6476d61a.png)

# Conclusions
With such high accuracy, recall, and precision, it is safe to say that this neural network model would serve as a competent supplementary tool for physicians, physician assistants, and nurses whose specialty may not be may be in brain tumor detection. It would help them more quickly and accurately detect brain tumors and flag scans which require further analysis by neurosurgeons, potentially giving them more time and energy to focus on other patients. These results together have the potential to improve health outcomes for patients in developing nations. 

## Further Steps
One of the most important future steps to be taken is deployment of this neural network model, perhaps in the form of a web app accessible to the relevant mdeical personell. Another very valuable step to be taken is to develop a multiclass classification neural network, which would be able to distinguish between the different main brain tumor types (glioma, meningioma, and pituitary).


## Information

- For information regarding the resorting of images from the kaggle dataset to be a binary classification problem, please see [this notebook](https://github.com/brooke57/BrainTumorImageClassification/blob/main/Supplemental_Notebooks/Renaming_Tumor_Images.ipynb), and please see the top of the [final notebook](https://github.com/brooke57/BrainTumorImageClassification/blob/main/Final_Binary_Brain_Tumor_Classification.ipynb) for details on how to load the data onto Google Colab.
- To see all models that were run, in addition to the ones in the final notebook, go to the ["All_Models_for_Binary_Brain_Tumor_Classification.ipynb](https://github.com/brooke57/BrainTumorImageClassification/blob/main/Supplemental_Notebooks/All_Models_for_Binary_Brain_Tumor_Classification.ipynb) notebook.
