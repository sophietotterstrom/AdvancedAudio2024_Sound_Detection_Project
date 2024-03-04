import os
import sys

import torch
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from model import Cnn14, init_layer

# These PANN's CNN Architectures are take from https://github.com/qiuqiangkong/audioset_tagging_cnn/tree/master:


class Transfer_Cnn14(nn.Module):
    def __init__(
            self, 
            sample_rate, 
            window_size, 
            hop_size, 
            mel_bins, 
            fmin,
            fmax, 
            classes_num, 
            freeze_base
        ):
        """
        Classifier for a new task using pretrained Cnn14 as a sub module.
        """

        super(Transfer_Cnn14, self).__init__()

        audioset_classes_num = 527

        self.base = Cnn14(
            sample_rate, 
            window_size, 
            hop_size, 
            mel_bins, 
            fmin,
            fmax, 
            audioset_classes_num
        )

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        
        checkpoint = torch.load(
            pretrained_checkpoint_path, 
            map_location=torch.device('cpu')
        )
        pretrained_dict = checkpoint

        model_dict = self.base.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        
        # 3. load the new state dict
        self.base.load_state_dict(model_dict)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)
        """

        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output = self.fc_transfer(embedding)

        return clipwise_output

