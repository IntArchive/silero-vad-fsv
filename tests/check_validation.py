from ..tuning.utils import validate
import torch
import numpy as np
import gc
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from omegaconf import OmegaConf
import torch.nn as nn
from ..tuning.utils import VADDecoderRNNJIT, SileroVadDataset

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(loader,
             jit_model,
             decoder,
             criterion,
             device):
    losses = AverageMeter()
    decoder.eval()

    predicts = []
    gts = []

    context_size = 64
    num_samples = 512
    stft_layer = jit_model._model.stft
    encoder_layer = jit_model._model.encoder

    with torch.no_grad():
        for _, (x, targets, masks) in tqdm(enumerate(loader), total=len(loader)):
            targets = targets.to(device)
            x = x.to(device)
            masks = masks.to(device)
            x = torch.nn.functional.pad(x, (context_size, 0))

            outs = []
       
            state = torch.zeros(0)
            for i in range(context_size, x.shape[1], num_samples):
                input_ = x[:, i-context_size:i+num_samples]
                out = stft_layer(input_)
                out = encoder_layer(out)
                out, state = decoder(out, state)
                outs.append(out)
            stacked = torch.cat(outs, dim=2).squeeze(1)

            predicts.extend(stacked[masks != 0].tolist())
            gts.extend(targets[masks != 0].tolist())

            loss = criterion(stacked, targets)
            loss = (loss * masks).mean()
            losses.update(loss.item(), masks.numel())
    score = roc_auc_score(gts, predicts)
    # Convert to numpy for easier inspection
    # predicts_np = np.asarray(predicts, dtype=float)
    # gts_np = np.asarray(gts, dtype=float)

    # # Debug / safety checks to avoid NaN ROC-AUC and to make the problem visible
    # if predicts_np.size == 0 or gts_np.size == 0:
    #     print('validate(): empty predicts or gts - cannot compute ROC-AUC')
    #     score = float('nan')
    # elif np.isnan(predicts_np).any() or np.isnan(gts_np).any():
    #     print('validate(): NaNs found in predicts or gts - cannot compute reliable ROC-AUC')
    #     print(f'  predicts has_nan={np.isnan(predicts_np).any()}, gts has_nan={np.isnan(gts_np).any()}')
    #     score = float('nan')
    # else:
    #     unique_classes = np.unique(gts_np)
    #     if unique_classes.size < 2:
    #         # This usually means that your validation annotations only contain a single class
    #         # (all-speech or all-non-speech). ROC-AUC is undefined in this case.
    #         print('validate(): only one class present in validation ground truth - ROC-AUC is undefined.')
    #         print(f'  unique classes in gts: {unique_classes}')
    #         score = float('nan')
    #     else:
    #         score = roc_auc_score(gts_np, predicts_np)

    torch.cuda.empty_cache()
    gc.collect()

    return losses.avg, round(score, 3)

if __name__ == '__main__':
    config = OmegaConf.load('./tuning/config.yml')
    dataset = SileroVadDataset(config, mode='val')
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    jit_model = torch.jit.load('./src/silero_vad/data/silero_vad.jit')
    decoder = VADDecoderRNNJIT()
    criterion = nn.BCELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    validate(loader, jit_model, decoder, criterion, device)