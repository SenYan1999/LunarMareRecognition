import numpy as np

def get_metric(criterion):
    if criterion == 'bce_dice_weight_loss':
        metric = {'Overall Loss': [], 'BCE Loss': [], 'Dice Loss': []}
        return metric
    else:
        raise(Exception('unsupported criterion %s' % criterion))

def update_metric(criterion, metric, metric_value):
    if criterion == 'bce_dice_weight_loss':
        metric['Overall Loss'].append(metric_value[0])
        metric['BCE Loss'].append(metric_value[1])
        metric['Dice Loss'].append(metric_value[2])
        return metric
    else:
        raise(Exception('unsupported criterion %s' % criterion))

def print_metric(criterion, metric, epoch, mode):
    if criterion == 'bce_dice_weight_loss':
        print('%s Epoch: %d | Overall Loss: %.2f | BCE Loss: %.2f | Dice Loss: %.2f' % (mode,
            epoch, np.mean(metric['Overall Loss']), np.mean(metric['BCE Loss']), np.mean(metric['Dice Loss'])))
    else:
        raise(Exception('unsupported criterion %s' % criterion))

def update_pbar(pbar, criterion, epoch, metric_value):
    if criterion == 'bce_dice_weight_loss':
        pbar.set_description('Epoch: %d | Overall Loss: %.2f | BCE Loss: %.2f | Dice Loss: %.2f' % (
            epoch, metric_value[0], metric_value[1], metric_value[2]))
    else:
        raise(Exception('unsupported criterion %s' % criterion))

    pbar.update(1)
