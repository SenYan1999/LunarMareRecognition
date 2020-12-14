import numpy as np

def get_metric(criterion):
    if criterion == 'general':
        metric = {'Acc': [], 'Acc_Class': [], 'mIoU': [], 'FWIoU': []}
        return metric
    elif criterion == 'bce_dice_weight_loss':
        metric = {'Overall Loss': [], 'BCE Loss': [], 'Dice Loss': []}
        return metric
    else:
        raise(Exception('unsupported criterion %s' % criterion))

def update_metric(criterion, metric, metric_value):
    if criterion == 'general':
        metric['Acc'].append(metric_value[0])
        metric['Acc_Class'].append(metric_value[1])
        metric['mIoU'].append(metric_value[2])
        metric['FWIoU'].append(metric_value[3])
        return metric
    elif criterion == 'bce_dice_weight_loss':
        metric['Overall Loss'].append(metric_value[0])
        metric['BCE Loss'].append(metric_value[1])
        metric['Dice Loss'].append(metric_value[2])
        return metric
    else:
        raise(Exception('unsupported criterion %s' % criterion))

def print_metric(criterion, metric, metric_general, epoch, mode):
    log_info = ''
    log_info += '%s Epoch: %d | Acc: %.2f | Acc_Class: %.2f | mIoU: %.2f | FWIoU %.2f' % (mode,
                                                                                        epoch,
                                                                                        metric_general['Acc'][-1],
                                                                                        metric_general['Acc_Class'][-1],
                                                                                        metric_general['mIoU'][-1],
                                                                                        metric_general['FWIoU'][-1])
    # all criterion need to log general metric
    if criterion == 'bce_dice_weight_loss':
        log_info += ' | Overall Loss: %.2f | BCE Loss: %.2f | Dice Loss: %.2f' % (np.mean(metric['Overall Loss']),
                                                                                  np.mean(metric['BCE Loss']),
                                                                                  np.mean(metric['Dice Loss']))
    else:
        raise(Exception('unsupported criterion %s' % criterion))

    print(log_info)

def get_loss_from_metric(criterion, metric):
    if criterion == 'bce_dice_weight_loss':
        return np.mean(metric['Overall Loss'])
    else:
        raise(Exception('unsupported criterion %s' % criterion))

def update_pbar(pbar, criterion, epoch, metric_value, metric_value_general):
    description = 'Epoch: %d | Acc: %.2f | Acc_Class: %.2f | mIoU: %.2f | FWIoU: %.2f' % (
        epoch, metric_value_general[0], metric_value_general[1], metric_value_general[2], metric_value_general[3])

    if criterion == 'bce_dice_weight_loss':
        description += ' | Overall Loss: %.2f | BCE Loss: %.2f | Dice Loss: %.2f' % (
        metric_value[0], metric_value[1], metric_value[2])
    else:
        raise(Exception('unsupported criterion %s' % criterion))

    pbar.set_description(description)
    pbar.update(1)

def update_train_experiment(experiment, criterion, metric_value, metric_value_general, step):
    experiment.log_metric('Acc', metric_value_general[0], step=step)
    experiment.log_metric('Acc_Class', metric_value_general[1], step=step)
    experiment.log_metric('mIoU', metric_value_general[2], step=step)
    experiment.log_metric('FWIoU', metric_value_general[3], step=step)

    if criterion == 'bce_dice_weight_loss':
        experiment.log_metric('Overall Loss', metric_value[0], step=step)
        experiment.log_metric('BCE Loss', metric_value[1], step=step)
        experiment.log_metric('Dice Loss', metric_value[2], step=step)
    else:
        raise(Exception('unsupported criterion %s' % criterion))

def update_dev_experiment(experiment, criterion, metric, metric_general, epoch):
    experiment.log_metric('Acc', metric_general['Acc'][-1], step=epoch)
    experiment.log_metric('Acc_Class', metric_general['Acc_Class'][-1], step=epoch)
    experiment.log_metric('mIoU', metric_general['mIoU'][-1], step=epoch)
    experiment.log_metric('FWIoU', metric_general['FWIoU'][-1], step=epoch)
    if criterion == 'bce_dice_weight_loss':
        experiment.log_metric('Overall Loss', np.mean(metric['Overall Loss']), step=epoch)
        experiment.log_metric('BCE Loss', np.mean(metric['BCE Loss']), step=epoch)
        experiment.log_metric('Dice Loss', np.mean(metric['Dice Loss']), step=epoch)
    else:
        raise(Exception('unsupported criterion %s' % criterion))
