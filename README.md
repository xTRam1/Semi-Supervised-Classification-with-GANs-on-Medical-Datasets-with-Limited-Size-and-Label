# Semi-Supervised-Classification-with-Generative-Adversarial-Networks-on-Medical-Datasets
Reproducible Software Artifact for paper "Semi-Supervised Classification with Generative Adversarial Networks on Medical Datasets with Limited Size and Label" by Erdoğan, 2020.

## Custom Training and Reproducing Results
1. Clone the repository
2. Navigate to the experiment you want to reproduce (ResNet50 or SS-DiffAugment-GAN.)
3.1 Modify ``` train.sh ``` for the experiment you want to produce, (Refer to paper for hyperparameters or try new ones.) and run it.
```sh
$ git clone https://github.com/xTRam1/Semi-Supervised-Classification-with-GANs-on-Medical-Datasets-with-Limited-Size-and-Label
$ cd Baseline-ResNet50 or SS-DiffAugment-GAN
$ bash train.sh  # be sure that you specify the path to your dataset
```
3.2 If you want to perform transfer learning from my results, select a checkpoint from the ```experiments``` folders and pass it as a "--load_path" argument to the shell script. An example is given below:
```sh
# For GAN training
python3 resImprovedGan.py --load_path "experiments/8/0.0002-0.0002/Model/checkpointbest1.pth" ... # Other arguments 

# For ResNet training
python3 train.py --load_path "experiments/Model/checkpointbest1.pth" ... # Other arguments
```
The path to an experiment result of GAN training is constructed as follows: "experiments/[batch size]/[discriminator learning rate/generator learning rate]/Model/[checkpoint file name (ends with .pth)]"

## Testing
1. Navigate to the experiment you want to reproduce (ResNet50 or SS-DiffAugment-GAN.)
2. Modify ``` test.sh ``` for the log you want to perform inference. (After training automatically you must have a file called logs containing the model checkpoint.), and run it.
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
