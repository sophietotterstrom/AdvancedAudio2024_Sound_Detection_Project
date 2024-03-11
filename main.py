"""
Main for loading the data, and training and evaluating the model.

Modified from https://github.com/mulimani/Sound-Event-Detection
"""

import sys

import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

import config
import sed_eval
from dcase_evaluate import get_SED_results, process_event_my
import model
from model_finetune import Transfer_Cnn14, Transfer_Cnn14_DecisionLevelMax
from utils import preprocess_data
from dataset_factory import BatchData, MelData


# Class labels of DCASE SED 2017 task - Events from street scene
CLASS_LABELS_DICT = {
    'brakes squeaking': 0,
    'car': 1,
    'children': 2,
    'large vehicle': 3,
    'people speaking': 4,
    'people walking': 5
}

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')


##############################################################################
################################ DATA LOADING ################################
##############################################################################
def load_data():

    # Development and evaluation sets paths
    dev_dir = config.development_dir
    eval_dir = config.evaluation_dir

    development_data = MelData(
        dev_dir, 
        CLASS_LABELS_DICT,
        sample_rate=config.sr,
        n_mels=config.nb_mel_bands,
        n_fft=config.nfft, 
        hop_length=config.hop_len
    )
    evaluation_data = MelData(
        eval_dir, 
        CLASS_LABELS_DICT,
        sample_rate=config.sr,
        n_mels=config.nb_mel_bands,
        n_fft=config.nfft, 
        hop_length=config.hop_len
    )

    X_dev, Y_dev = development_data.mel_tensor, development_data.label_tensor
    X_eval, Y_eval = evaluation_data.mel_tensor, evaluation_data.label_tensor

    X_dev, Y_dev, X_eval, Y_eval = preprocess_data(X_dev, Y_dev, X_eval, Y_eval, config.seq_length)
    # X_dev, X_eval = torch.from_numpy(X_dev).float(), torch.from_numpy(X_eval).float()

    train_loader = DataLoader(
        BatchData(X_dev, Y_dev), 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers
    )
    test_loader = DataLoader(
        BatchData(X_eval, Y_eval), 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers
    )

    return train_loader, test_loader

def load_plot_data():
    # Development and evaluation sets paths
    plot_dir = '../dataset/PlotSet/'

    plot_data = MelData(
        plot_dir, 
        CLASS_LABELS_DICT,
        sample_rate=config.sr,
        n_mels=config.nb_mel_bands,
        n_fft=config.nfft, 
        hop_length=config.hop_len
    )

    X_dev, Y_dev = plot_data.mel_tensor, plot_data.label_tensor
    X_eval, Y_eval = plot_data.mel_tensor, plot_data.label_tensor

    X_dev, Y_dev, X_eval, Y_eval = preprocess_data(X_dev, Y_dev, X_eval, Y_eval, config.seq_length)
    # X_dev, X_eval = torch.from_numpy(X_dev).float(), torch.from_numpy(X_eval).float()

    plot_loader = DataLoader(
        BatchData(X_dev, Y_dev), 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers
    )
    return plot_loader


##############################################################################
################################ MODEL FUNCTS ################################
##############################################################################
def train(model, train_loader, epochs, check_point):
    step = 0
    model.to(device)

    criteria = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.001)

    for epoch_idx in range(1, epochs + 1):
        model.train()
        sum_loss = 0

        for batch_idx, (mel, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # unpack batch and transfer to device
            mel = mel.to(device)
            target = target.to(device).float()

            # fetch predictions
            preds = torch.sigmoid(model(mel)) # NOTE before: model(mel)

            loss = criteria(preds, target)
            sum_loss += loss.item()

            # backpropagate and update weights
            loss.backward()
            optimizer.step()

            step += 1

            if (batch_idx + 1) % check_point == 0 or (batch_idx + 1) == len(train_loader):                
                print(f'Epoch: {epoch_idx:03d} | '
                      f'Batch: {batch_idx + 1:03d} | '
                      f'Step: {step:06d} | '
                      f'Train Loss {sum_loss / (batch_idx + 1):7.4f}')
        
        scheduler.step()

def evaluate(model, test_loader):

    model.to(device)
    model.eval()

    preds_list = []
    target_list = []

    for batch_idx, (mel, target) in enumerate(test_loader):
        # unpack batch and transfer to device
        mel, target = mel.to(device), target.to(device).float()

        preds = model(mel)
        
        # append predictions and targets
        # preds_list.extend(preds.cpu().detach().numpy())
        preds_list.extend(preds.view(-1, preds.size(2)).cpu().detach().numpy())
        target_list.extend(target.view(-1, target.size(2)).cpu().detach().numpy())

    # display evaluation metrics
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=list(CLASS_LABELS_DICT.keys()),
        time_resolution=1.0
    )
    output, test_ER, test_F1, class_wise_metrics = get_SED_results(
        np.array(target_list), 
        np.array(preds_list),
        list(CLASS_LABELS_DICT.keys()), 
        segment_based_metrics,
        threshold=0.5,
        hop_size=config.hop_len, 
        sample_rate=config.sr
    )
    print(output)
    print(f'F1: {test_F1:.3f} | '
          f'ER: {test_ER:.3f}')


########################################################################################
# Mikhail's code from https://github.com/msilaev/Sound-Event-Detection-course-project/
# to ensure plots for the report match each other
########################################################################################

check_time_stamp_folder = "../dataset/CheckTimeStamps/"
time_stamp_predict_file = "time_stamp_predict.txt"
time_stamp_label_file = "time_stamp_label.txt"

def predict_time_stamps(model, usage_loader, check_time_stamp_folder):
    model.to(device)
    model.eval()

    preds_list = []
    target_list = []

    for batch_idx, (mel, target) in enumerate(usage_loader):
        mel, target = mel.to(device), target.to(device).float()
        preds = model(mel)

        preds_list.extend(preds.view(-1, preds.size(2)).cpu().detach().numpy())
        target_list.extend(target.view(-1, preds.size(2)).cpu().detach().numpy())

    hop_length_seconds = config.hop_len/config.sr
    threshold = 0.5

    # this is needed to generate  time stamps of predicted events
    process_event_my(list(CLASS_LABELS_DICT.keys()),
                  np.array(preds_list).T,
                  threshold, hop_length_seconds, time_stamp_predict_file)

    process_event_my(list(CLASS_LABELS_DICT.keys()),
                     np.array(target_list).T,
                     threshold, hop_length_seconds, time_stamp_label_file)


def plot_sound_events():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    time_stamp_predict_file = "time_stamp_predict.txt"
    time_stamp_label_file = "time_stamp_label.txt"

    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors for different classes
    class_colors_dict = {
        'brakes squeaking': '#E41A1C',  # Red
        'car': '#377EB8',  # Blue
        'children': '#4DAF4A',  # Green
        'large vehicle': '#984EA3',  # Purple
        'people speaking': '#FF7F00',  # Orange
        'people walking': '#FFFF33'  # Yellow
    }

    column_names = ["start_time", "end_time", "class"]
    # Load the dataset
    data = pd.read_csv(time_stamp_predict_file, header=None, names=column_names)

    data_annotation = pd.read_csv(time_stamp_label_file, header=None, names=column_names)

    settings = {'annotated': {'y_value': 1, 'linestyle': '-', 'label': 'Annotated'},
                'model_output': {'y_value': 0.995, 'linestyle': '--', 'label': 'Model Output'}}
    # Closer y_value and different linestyle

    # Plot each class activity
    for _, row in data.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        audio_class = row['class']
        color = class_colors_dict[audio_class]

        ax.plot([start_time, end_time], [settings['annotated']['y_value'], settings['annotated']['y_value']],
                color=color,
                linewidth=10, linestyle=settings['annotated']['linestyle'])

    for _, row in data_annotation.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        audio_class = row['class']
        color = class_colors_dict[audio_class]

        ax.plot([start_time, end_time], [settings['model_output']['y_value'], settings['model_output']['y_value']],
                color=color,
                linewidth=10, linestyle=settings['model_output']['linestyle'])

        # Create legends
    class_legend = [mlines.Line2D([], [], color=color, label=audio_class) for audio_class, color in
                    class_colors_dict.items()]

    # data_legend = [mlines.Line2D([], [], color='black', linestyle=settings[data_type]['linestyle'],
    #                                 label=settings[data_type]['label']) for data_type in settings]

    # Add legends to the plot
    first_legend = ax.legend(handles=class_legend, title="Classes", loc='upper left')
    ax.add_artist(first_legend)
    # ax.legend(handles=data_legend, title="Data Type", loc='upper right')

    ax.set_yticks([])  # Hide Y axis
    ax.set_xlabel('Time (s)')
    ax.set_title('Audio Class Activity Over Time')
    ax.set_ylim(0.98, 1.02)

    plt.tight_layout()
    plt.savefig("ActivityCNN14.png")
    plt.show()
########################################################################################
# Mikhail's code end
########################################################################################


def main():

    np.random.seed(1900)

    model = Transfer_Cnn14_DecisionLevelMax(
        sample_rate=config.sr, 
        window_size=config.win_len, 
        hop_size=config.hop_len,
        mel_bins=config.nb_mel_bands,
        fmin=config.fmin,
        fmax=config.fmax, 
        classes_num=config.num_classes, 
        freeze_base=False
    )
    model = model.to(device)
    model.load_from_pretrain(
        pretrained_checkpoint_path=config.pretrained_checkpoint_path
    )

    plot = False
    if plot:
        plot_loader = load_plot_data()
        evaluate(model, plot_loader)
        predict_time_stamps(model, plot_loader, check_time_stamp_folder)

        plot_sound_events()
        sys.exit()
        
    # fetch training and test dataloaders
    train_loader, test_loader = load_data()

    #model.train()
    #train(model, train_loader, epochs=config.epochs, check_point=config.check_point)
    
    evaluate(model, test_loader)

if __name__ == '__main__':
    main()
