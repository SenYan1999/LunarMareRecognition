from .metric import *
from tqdm import tqdm

try:
    from apex import amp
except:
    print('Apex not installed in this environment')

def train_epoch(model, data_dl, optimizer, epoch, criterion, experiment, args):
    model.train()
    metric = get_metric(args.criterion)

    with tqdm(total=len(data_dl)) as pbar:
        for batch in data_dl:
            input, label = map(lambda x: x.to(args.device), batch)

            logits = model(input)
            loss, metric_value = criterion(logits, label.unsqueeze(dim=1).float())

            # update parameters
            optimizer.zero_grad()
            if args.fp16:
                with amp.scaled_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # update log
            metric = update_metric(args.criterion, metric, metric_value)
            update_pbar(pbar, args.criterion, epoch, metric_value)

    return metric
