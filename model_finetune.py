"""
Loading a pretrained model.

Modified from https://github.com/mulimani/Acoustic-Scene-Classification/
These PANN's CNN Architectures are take from https://github.com/qiuqiangkong/audioset_tagging_cnn/tree/master:
"""

import torch
torch.manual_seed(0)
from torch.nn import Linear, Module

from model import Cnn14, init_layer, Cnn14_DecisionLevelMax


class Transfer_Cnn14_DecisionLevelMax(Module):
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

        super(Transfer_Cnn14_DecisionLevelMax, self).__init__()

        #audioset_classes_num = 527

        self.base =  Cnn14_DecisionLevelMax(
            sample_rate, 
            window_size, 
            hop_size, 
            mel_bins, 
            fmin,
            fmax, 
            classes_num #audioset_classes_num
        )

        # Transfer to another task layer
        self.fc_transfer = Linear(2048, classes_num, bias=True)

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
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in model_dict
        }
        
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        
        # 3. load the new state dict
        self.base.load_state_dict(model_dict)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)
        """

        output_dict = self.base(input, mixup_lambda)

        framewise = output_dict['framewise_output']
        clipwise = output_dict['clipwise_output']
        embedding = output_dict['embedding']
        
        #embedding = output_dict['embedding']
        #embedding_clipwise_output = self.fc_transfer(embedding)

        return framewise


class Transfer_Cnn14(Module):
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

        #audioset_classes_num = 527

        self.base = Cnn14(
            sample_rate, 
            window_size, 
            hop_size, 
            mel_bins, 
            fmin,
            fmax, 
            classes_num #audioset_classes_num
        )

        # Transfer to another task layer
        self.fc_transfer = Linear(2048, classes_num, bias=True)

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
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in model_dict
        }
        
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

