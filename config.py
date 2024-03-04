# Mel spectrogram parameters

nfft = 2048
win_len = nfft
hop_len = win_len // 2
nb_mel_bands = 40
sr = 44100


# Sequence parameters

seq_length = 256
nb_channels = 1

# Dataset params
development_dir = '../dataset/SED_2017_street/TUT-sound-events-2017-development/'
evaluation_dir = '../dataset/SED_2017_street/TUT-sound-events-2017-evaluation/'