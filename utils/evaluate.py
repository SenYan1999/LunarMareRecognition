import torch
import os
import tifffile as tiff
import json

from .log import *
from .metric import Evaluator
from tqdm import tqdm
from functools import reduce

all_pixel_error = []

def evaluate_model(model, dev_dl, criterion, args, output_dir=None):
    model.eval()
    general_metric = get_metric('general')
    metric = get_metric(args.criterion)

    if output_dir != None:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    # load transform information
    normalize_info = torch.load(args.data_normalize_info)
    mean, std = normalize_info['mean'].item(), normalize_info['std'].item()

    evaluator = Evaluator(2)
    idx_begin = 0
    with torch.no_grad():
        for batch in tqdm(dev_dl, total=len(dev_dl)):
            input, label = map(lambda x: x.to(args.device), batch)

            logits = model(input)
            loss, metric_value = criterion(logits, label.unsqueeze(dim=1).float())
            general_metric_value = evaluator.add_batch(label.unsqueeze(dim=1).cpu().numpy(), logits.detach().cpu().numpy())

            # update metric
            metric = update_metric(args.criterion, metric, metric_value)
            general_metric = update_metric('general', general_metric, general_metric_value)

            if output_dir:
                write_img_batch(logits, output_dir, idx_begin, labels=label, origin_image=input, threshold=args.output_threshold, transform_info=(mean, std))
                idx_begin += input.shape[0]

    with open(os.path.join('analysis', args.analysis_pe), 'w') as f:
        json.dump(all_pixel_error, f)

    return metric, general_metric

def convert_logit_to_img_sigmoid(logit):
    pred = torch.sigmoid(logit.squeeze()).detach().cpu()
    assert len(pred.shape) == 2

    pred[torch.where(pred >= 0.5)] = 255
    pred[torch.where(pred < 0.5)] = 0

    pred = np.array(pred).astype(np.uint8)

    return pred

def write_img_batch(logits, out_dir, idx_begin, labels=None, origin_image=None, threshold=None, transform_info=None):
    for i in range(logits.shape[0]):
        pred_img = convert_logit_to_img_sigmoid(logits[i])
        label = labels[i].squeeze().detach().cpu().numpy().astype(np.uint8)
        origin = (origin_image[i].squeeze().detach().cpu().numpy() * transform_info[1] + transform_info[0]).astype(np.uint8)
        label[np.where(label >= 0.5)] = 255

        pixel_error = np.sum(pred_img != label) / reduce(lambda x, y: x * y, pred_img.shape)
        all_pixel_error.append(pixel_error)
        if pixel_error > threshold:
            tiff.imwrite(os.path.join(out_dir, '%s_pred.tif' % str(i+idx_begin)), pred_img)
            tiff.imwrite(os.path.join(out_dir, '%s_truth.tif' % str(i+idx_begin)), label)
            origin = origin
            tiff.imwrite(os.path.join(out_dir, '%s_ground_truth.tif' % str(i+idx_begin)), origin)

