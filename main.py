from comet_ml import OfflineExperiment

import torchvision
import h5py
import utils
import sys
import os
import torch.distributed as dist

from tqdm import tqdm
from model import UNet
from args import args
from utils.loss_fn import *

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
    import apex
except:
    print('Apex not installed in this environment')


log_file = os.path.join('log/log_SGD_common_augmentation.log')
log_f = open(log_file, 'w', encoding='utf-8')
sys.stdout = log_f

### TODO: Data Augmentation
### TODO: Remove High Latitute
### TODO: Different Optimizer
### TODO: Different Loss Func
### TODO: Different Model
### TODO: Different Data Transform
### TODO: Log to File
### TODO: FIX NUM_WORKERS 1 FOR DEV_DL DUE TO H5PY

def prepare_data():
    # # write raw data
    utils.prepare_data(args.raw_data_dir, args.data)

    # split dataset into train data and dev data
    data = h5py.File(args.data, 'r')
    train_num = int(args.train_data_ratio * data['input'].shape[0])
    dev_num = data['input'].shape[0] - train_num
    train_data, dev_data = utils.split_dataset(data, train_num, dev_num)

    # save train h5
    train_h5 = h5py.File(args.train_data, 'w')
    train_h5.create_dataset('input', data=train_data[0])
    train_h5.create_dataset('label', data=train_data[1])
    train_h5.close()

    # save dev h5
    dev_h5 = h5py.File(args.dev_data, 'w')
    dev_h5.create_dataset('input', data=dev_data[0])
    dev_h5.create_dataset('label', data=dev_data[1])
    dev_h5.close()

    # get the normalize information
    train_data = utils.LunarDataset(args.train_data)
    mean = 0.
    std = 0.
    for i in tqdm(range(len(train_data))):
        input = train_data[i][0].reshape(1, 1, -1)
        mean += input.mean(2).sum(0)
        std += input.std(2).sum(0)
    mean /= len(train_data)
    std /= len(train_data)
    torch.save({'mean': torch.FloatTensor([mean]).reshape(-1), 'std': torch.FloatTensor([std]).reshape(-1)},
               args.data_normalize_info)

def train_and_evaluate(experiment):
    # prepare distributed train
    if args.distributed:
        assert args.fp16 # we only allow distributed training using apex
        # dist.init_process_group(backend=args.backend, init_method='tcp://127.0.0.1:4318', rank=args.local_rank, world_size=2)
        dist.init_process_group(backend=args.backend)
        torch.cuda.set_device(args.local_rank)
        print(f'Now local rank is {args.local_rank}')

    # get the device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # retrieve data
    print('Loading the datasets...')
    normalize_info = torch.load(args.data_normalize_info)
    transform = torchvision.transforms.Normalize(normalize_info['mean'], normalize_info['std'])
    train_data, dev_data = utils.LunarDataset(args.train_data, transform=transform, augmentation=utils.common_seq), utils.LunarDataset(args.dev_data, transform=transform)
    print('- done.')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data) if args.distributed else None
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                           pin_memory=True, sampler=train_sampler)
    dev_dl = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                         pin_memory=True, sampler=dev_sampler)

    # define model
    model = UNet(n_channels=1, n_classes=args.num_classes, bilinear=False).to(args.device)

    # define optimizer and scheduler
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if args.num_classes > 1 else 'max', patience=2)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        if args.distributed:
            model = DDP(model)

    # get the criterion
    criterion = globals()[args.criterion]

    min_loss = 1e8
    for epoch in range(args.epochs):
        with experiment.train():
            train_metric, train_metric_general = utils.train_epoch(model, train_dl, optimizer, epoch, criterion, experiment, args)
        output_dir = None if epoch % 20 != 0 else args.pred_img_out_dir
        dev_metric, dev_metric_general = utils.evaluate_model(model, dev_dl, criterion, args, output_dir=output_dir)

        with experiment.test():
            utils.update_dev_experiment(experiment, args.criterion, dev_metric, dev_metric_general, epoch)

        utils.print_metric(args.criterion, train_metric, train_metric_general, epoch, mode='Train')
        utils.print_metric(args.criterion, dev_metric, dev_metric_general, epoch, mode='Test')

        # adjust learning rate
        loss = utils.get_loss_from_metric(args.criterion, dev_metric)
        scheduler.step(loss)

        # update min_loss
        is_min = min_loss > loss
        min_loss = min(min_loss, loss)

        if args.local_rank == 0:
            print('Min Loss: %.2f' % min_loss)
        if is_min:
            torch.save({'state_dict': model.state_dict(), 'min_loss': min_loss}, args.save_model)

def evaluate():
    # firstly we need to load the model
    pretrained_model = args.save_model
    checkpoint = torch.load(pretrained_model, map_location=lambda storage, loc: storage.cuda(args.local_rank))

    # define model
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=args.num_classes, bilinear=False).to(args.device)
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])

    # prepare dev dataloader
    print('Loading the datasets...')
    normalize_info = torch.load(args.data_normalize_info)
    transform = torchvision.transforms.Normalize(normalize_info['mean'], normalize_info['std'])
    dev_data = utils.LunarDataset(args.dev_data, transform=transform)
    dev_dl = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                         pin_memory=True, sampler=None)
    print('- done.')
    print('total dev samples: %d' % len(dev_data))

    # evaluate
    criterion = globals()[args.criterion]
    metric, general_metric = utils.evaluate_model(model, dev_dl, criterion, args, output_dir=args.pred_img_out_dir)
    utils.print_metric(args.criterion, metric, general_metric, 0, 'Dev')

def main():
    if args.prepare_data:
        prepare_data()

    if args.train:
        if args.local_rank == 0:
            experiment = OfflineExperiment(
                api_key="vmCRwu7hHUSAp05U6DDME06pR",
                project_name="lunarmarerecognization",
                workspace="senyan1999",
                offline_directory='comet_experiment',
            )
            experiment.add_tag('UNet')
            experiment.log_parameters({
                'model': 'UNet',
                'lr': args.lr,
                'criterion': args.criterion
            })
        else:
            experiment = OfflineExperiment(offline_directory='comet_experiment', disabled=True)

        train_and_evaluate(experiment)

    if args.evaluate:
        evaluate()

if __name__ == '__main__':
    main()
