# SVMtextureClassifier

A MATLAB code is provided for a SVM texture classifier to differentiate diseased marrows from normal marrows. 

The code involves the following steps. 

1) Open a DICOM file (sagittal, T1-weighted spine images) and normalize marrow signals using disks
2) Marrow segmentation using a 3D GrowCut algorithm
3) Feature extraction 
4) Data preparation for SVM 
5) Two-class SVM using libsvm  
