"""

Modified from https://github.com/mulimani/Sound-Event-Detection

"""

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import config
import sed_eval
from dcase_evaluate import get_SED_results
import model
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
else:
    device = torch.device('cpu')


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
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        BatchData(X_eval, Y_eval), 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    return train_loader, test_loader


def train(model, train_loader, epoch, check_point):
    step = 0
    model.to(device)
    criteria = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0.001)

    for epoch_idx in range(1, epoch + 1):
        model.train()
        sum_loss = 0
        for batch_idx, (mel, target) in enumerate(train_loader):
            optimizer.zero_grad()
            mel, target = mel.to(device), target.to(device).float()
            logits = model(mel)
            loss = criteria(logits, target)
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
            step += 1

            if (batch_idx + 1) % check_point == 0 or (batch_idx + 1) == len(train_loader):
                print('==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.3f}'.
                      format(epoch_idx, batch_idx + 1, step, sum_loss / (batch_idx + 1)))

        scheduler.step()


def evaluate(model, test_loader):
    model.to(device)
    model.eval()

    preds_list = []
    target_list = []

    for batch_idx, (mel, target) in enumerate(test_loader):
        mel, target = mel.to(device), target.to(device).float()
        preds = torch.sigmoid(model(mel))

        preds_list.extend(preds.view(-1, preds.size(2)).cpu().detach().numpy())
        target_list.extend(target.view(-1, target.size(2)).cpu().detach().numpy())

    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=list(CLASS_LABELS_DICT.keys()),
        time_resolution=1.0
    )

    output, test_ER, test_F1, class_wise_metrics = get_SED_results(np.array(target_list), np.array(preds_list),
                                                                   list(__class_labels_dict.keys()), segment_based_metrics,
                                                                   threshold=0.5,
                                                                   hop_size=config.hop_len, sample_rate=config.sr)
    print(output)
    print('F1: {:.3f}, ER: {:.3f}'.format(test_F1, test_ER))


if __name__ == '__main__':

    np.random.seed(1900)
    model = model.CRNN(classes_num=6).to(device)

    # fetch training and test dataloaders
    train_loader, test_loader = load_data()

    train(model, train_loader, epoch=config.epochs, check_point=config.check_point)
    evaluate(model, test_loader)
