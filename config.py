"""
Pretrained model downloaded from https://zenodo.org/records/3987831 [1]
[1] Kong, Qiuqiang, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, 
    and Mark D. Plumbley. "Panns: Large-scale pretrained audio neural 
    networks for audio pattern recognition." 
    arXiv preprint arXiv:1912.10211 (2019).
"""

#################################
# Mel spectrogram parameters
#################################
nfft = 2048
win_len = nfft
hop_len = win_len // 2
nb_mel_bands = 40
sr = 44100

fmin=50
fmax=14000

#################################
# Sequence parameters
#################################
seq_length = 256
nb_channels = 1

#################################
# Dataset params
#################################
development_dir = '../dataset/SED_2017_street/TUT-sound-events-2017-development/'
#evaluation_dir = '../dataset/SED_2017_street/TUT-sound-events-2017-evaluation/'
evaluation_dir = '../dataset/MyEvaluationSet/'
num_classes = 6

#################################
# Model parameters
#################################
#pretrained_checkpoint_path = '../Cnn14_mAP=0.431.pth'
pretrained_checkpoint_path = '../Cnn14_DecisionLevelMax_mAP=0.385.pth'

lr = 0.0001         # learning rate for training
epochs = 100        # total number of training epochs
batch_size = 32     # mini bath-size
num_workers = 0     # numbers parallel workers to use
check_point = 50    # check point