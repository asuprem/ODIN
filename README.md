# ODIN
This repo contains the code for 'ODIN: Automated Drift Detection and Recovery in Video Analytics'. It contains two code bases -- training for the DA-GAN, and the ODIN system itself.

### Requirements
You will need: 
  - PyTorch 1.4.0 
  - Torchvision 0.4.0 
  - kaptan 0.5.12

Multi-GPU training is currently not supported. You will need to ensure torch recognizes only one GPU, otherwise several functions will throw NotImplementedError().

### DA-GAN training

The DA-GAN can be trained from within DA_GAN.ipynb. The code will download CIFAR-10 and train the DA-GAN model. We designed it to work with Google Colab, so you will need to upload the `vaegan.zip` file into your colab session and unzip it within the DA_GAN.ipynb notebook (first code cell). The completed models should be backed up to your google drive.


### ODIN System

`mlep_odin_main` is code for ODIN itself. Specifically, the `mlep` folder contains classes for KL-based Delta Band distribution detector (ZonedDistribution class) plus the odin system (MLEPDriftAdaptor class). Inside the directory, you can install it as a development package:

    cd mlep
    pip3 install -e .
 
This is the most up to date set of instructions. However, we are continuously updating the codebase, so the most recent commit may break changes. We'll try to update readme whenever this happens.

You should install the minimum (or standard) requirements with pip from the `requirements.txt` or `minrequirements.txt`. You will also need to install Pytorch and Torchvision. ODIN has out-of-the-box support for image and text embedding methods: DA-GAN (in `mlep.models` for images) and word2vec for text. The ODIN system is described in `mlep.core.MLEPModedlDriftAdaptor`. You will need to instantiate this class for an application and pass in a configuration based on `mlep/configuration/MLEPModelDriftAdaptorConfiguration.py`


