# Semi-Supervised-Classification-with-Generative-Adversarial-Networks-on-Medical-Datasets
Reproducible Software Artifact for paper "Semi-Supervised Classification with Generative Adversarial Networks on Medical Datasets with Limited Size and Label" by Erdoğan, 2020.

Note: Due to Github data storage quotas, I couldn't upload my experiment results. If you want to take my saved models, tensorboard logs, sample generated fake images, and fixed z generated images, for both the baseline and the GAN, please use these commands below to download them to your environment. If not, if you just want to perform custom training, you can skip this part.
```sh
$ git clone https://github.com/xTRam1/Semi-Supervised-Classification-with-GANs-on-Medical-Datasets-with-Limited-Size-and-Label

# For the ResNet Baseline
$ cd Baseline-ResNet50
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BHvjoytWenLllyTcEmzvCxV-QER2-74M' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BHvjoytWenLllyTcEmzvCxV-QER2-74M" -O experiments_resnet.zip && rm -rf /tmp/cookies.txt
$ unzip experiments_resnet.zip && rm -rf experiments_resnet.zip

# For the GAN
$ cd SS-DiffAugment-GAN
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GWlzt6xkD8Z4tNY0ioodie26jmeYeV1P' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GWlzt6xkD8Z4tNY0ioodie26jmeYeV1P" -O experiments_gan.zip && rm -rf /tmp/cookies.txt
$ unzip experiments_gan.zip && rm -rf experiments_gan.zip
```

## Custom Training and Reproducing Results
1. Clone the repository
2. Navigate to the experiment you want to reproduce (ResNet50 or SS-DiffAugment-GAN.)
3. Modify ``` train.sh ``` for the experiment you want to produce, (Refer to paper for hyperparameters or try new ones.) and run it.
```sh
$ git clone https://github.com/xTRam1/Semi-Supervised-Classification-with-GANs-on-Medical-Datasets-with-Limited-Size-and-Label
$ cd Baseline-ResNet50 or SS-DiffAugment-GAN
$ bash train.sh  # be sure that you specify the path to your dataset
```
4. If you want to perform transfer learning from my results, select a checkpoint from the ```experiments``` folders and pass it as a "--load_path" argument to the shell script. An example is given below:
```sh
# For GAN training
python3 resImprovedGan.py --load_path "experiments/8/0.0002-0.0002/Model/checkpointbest1.pth" ... # Other arguments 

# For ResNet training
python3 train.py --load_path "experiments/Model/checkpointbest1.pth" ... # Other arguments
```
The path to an experiment result of GAN training is constructed as follows: ```experiments/[batch size]/[discriminator learning rate/generator learning rate]/Model/[checkpoint file name (ends with .pth)]```

5. After or during training you can track your model's performance by using Tensorboard. You will have a ```logs``` folder created for you during training which Tensorboard will use to plot accuracy and learning rate plots for your model. The below command runs tensorboard:
```
$ tensorboard --logdir logs # Make sure you have Tensorboard installed 
```

## Testing
1. Navigate to the experiment you want to reproduce (ResNet50 or SS-DiffAugment-GAN.)
2. Modify ``` test.sh ``` for the log you want to perform inference. (After training automatically you must have a file called logs containing the model checkpoint - use the above template to access a single checkpoint in order to test its accuracy), and run it.
```sh
$ git clone https://github.com/xTRam1/Semi-Supervised-Classification-with-GANs-on-Medical-Datasets-with-Limited-Size-and-Label
$ cd Baseline-ResNet50 or SS-DiffAugment-GAN
$ bash test.sh  # be sure that you specify the path to your dataset
```

## Experiment Results of GAN training
In the GAN folder, inside the ```experiments``` folder, you can find sample images generated from random latent vectors by the generator at specified iterations during the training. You can also find the images generated from a fixed latent vector during the training process, showing how well the generator adapts to the task. 
```
--> SS-DiffAugment-GAN
  --> experiments
    --> 8
      --> Fixed Z
        --> Image 1
        --> Image 2
        ...
      --> Samples
        --> Image 1
        --> Image 2
      ...
    --> 16
    --> 32
```

Note: There is no need to download a dataset as it is already on the repo. However, you need to specify the path of the dataset images to ``` train.sh ``` and ``` test.sh ``` before training and testing respectively.
