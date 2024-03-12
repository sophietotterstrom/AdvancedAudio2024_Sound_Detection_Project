# **Sound Event Detection using PANNs**

This repository contains the the code for [Sound Event Detection](https://dcase.community/challenge2017/task-sound-event-detection-in-real-life-audio) written using PyTorch using a convolutional neural network PANN (PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition).

Performance of the model is evaluated using [sed_eval](https://tut-arg.github.io/sed_eval/) and [dcase_util](https://github.com/DCASE-REPO/dcase_util) toolboxes.

### Prerequisite

* torch>=1.11.0
* numpy>=1.19.5
* pandas>=1.3.4
* torchaudio>=0.11.0
* scikit-learn>=0.24.2
* sed_eval>=0.1.8
* dcase_util>=0.2.16


### Getting started

1. Install the requirements: pip install -r requirements.txt
2. Download the DCASE 2017 Task3 [development](https://zenodo.org/records/814831) and [evaluation](https://zenodo.org/records/1040179) datasets into  dataset/SED_2017_street folder.
3. Downaload our own evaluation dataset from [GoogleDrive](https://drive.google.com/file/d/1oxUgjIjj3x8ThobMQKE7SoMyF4bJqhdI/view).
4. Change configuration parameters ('config.py') and boolenans in the 'main' function to change around what specific task you want to do (load pretrained, train from scratch, plot an example file).
6. Run experiment: python main.py


### Acknowledgement 

This code is built on [dcase_util](https://github.com/DCASE-REPO/dcase_util) toolbox.

Code uses the following repos:
* [PANNs: audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn/tree/master/pytorch) 
* [Sound-Event-Detection](https://github.com/mulimani/Sound-Event-Detection)
* [Acoustic-Scene-Classification](https://github.com/mulimani/Acoustic-Scene-Classification)
