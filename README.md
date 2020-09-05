# Semi-Supervised-Classification-with-Generative-Adversarial-Networks-on-Medical-Datasets
Reproducible Software Artifact for paper "Semi-Supervised Classification with Generative Adversarial Networks on Medical Datasets with Limited Size and Label" by Erdoğan, 2020.

# Reproducing Results
1. Navigate to the experiment you want to reproduce (ResNet50 or SS-DiffAugment-GAN.)
2. Modify ''' 
train.sh 
''' for the experiment you want to produce. (Refer to paper for hyperparameters.)
3. Write ''' 
bash train.sh 
''' to the command line to train. 

# Testing
1. Navigate to the experiment you want to reproduce (ResNet50 or SS-DiffAugment-GAN.)
2. Modify ''' 
test.sh 
''' for the log you want to perform inference. (After training automatically you must have a file called logs containing the model checkpoint.)
3. Write ''' 
bash test.sh 
''' to the comman line to test. 

Note: There is no need to download a dataset as it is already on the repo. However, you need to specify the path of the dataset images to ''' 
train.sh 
''' and ''' 
test.sh 
''' before training and testing respectively.
