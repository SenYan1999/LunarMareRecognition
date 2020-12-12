import torch
import os
import numpy as np
import tifffile as tiff

from .metric import *

def evaluate_model(model, dev_dl, criterion, args, output_dir=None):
    model.train()
    metric = get_metric(args.criterion)

    with torch.no_grad():
        for batch in dev_dl:
            input, label = map(lambda x: x.to(args.device), batch)

            logits = model(input)
            loss, metric_value = criterion(logits, label.unsqueeze(dim=1).float())

            # update metric
            metric = update_metric(args.criterion, metric, metric_value)

            if output_dir:
                write_img_batch(logits, output_dir, 0, labels=label)

    return metric

def convert_logit_to_img_sigmoid(logit):
    pred = torch.sigmoid(logit.squeeze()).detach().cpu()
    assert len(pred.shape) == 2

    pred[torch.where(pred >= 0.5)] = 255
    pred[torch.where(pred < 0.5)] = 0

    pred = np.array(pred).astype(np.uint8)

    return pred

def write_img_batch(logits, out_dir, idx_begin, labels=None):
    for i in range(logits.shape[0]):
        pred_img = convert_logit_to_img_sigmoid(logits[i])
        tiff.imwrite(os.path.join(out_dir, '%s_pred.tif' % str(i+idx_begin)), pred_img)

        if type(labels) != type(None):
            label = labels[i].squeeze().detach().cpu().numpy().astype(np.uint8)
            label[np.where(label >= 0.5)] = 255
            tiff.imwrite(os.path.join(out_dir, '%s_truth.tif' % str(i+idx_begin)), label)
