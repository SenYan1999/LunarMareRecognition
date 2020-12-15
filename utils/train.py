from .log import *
from .metric import Evaluator
from tqdm import tqdm

try:
    from apex import amp
except:
    pass

def train_epoch(model, data_dl, optimizer, epoch, criterion, experiment, args):
    model.train()
    general_metric = get_metric('general')
    metric = get_metric(args.criterion)

    evaluator = Evaluator(2)
    with tqdm(total=len(data_dl)) as pbar:
        for i, batch in enumerate(data_dl):
            input, label = map(lambda x: x.to(args.device), batch)

            logits = model(input)
            loss, metric_value = criterion(logits, label.unsqueeze(dim=1).float())
            general_metric_value = evaluator.add_batch(label.unsqueeze(dim=1).cpu().numpy(), logits.detach().cpu().numpy())

            # update parameters
            optimizer.zero_grad()
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # update log
            metric = update_metric(args.criterion, metric, metric_value)
            general_metric = update_metric('general', general_metric, general_metric_value)

            if args.local_rank == 0:
                step = i + epoch * len(data_dl)
                update_pbar(pbar, args.criterion, epoch, metric_value, general_metric_value)
                update_train_experiment(experiment, args.criterion, metric_value, general_metric_value, step)

    return metric, general_metric
