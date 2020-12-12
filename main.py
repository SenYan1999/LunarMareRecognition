import torch
import torchvision
import h5py
import utils

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

### TODO: Data Augmentation
### TODO: Different Loss Func
### TODO: Different Model
### TODO: Different Data Transform
### TODO: Log to File

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

def train_and_evaluate():
    # prepare distributed train
    if args.distributed:
        assert args.fp16 # we only allow distributed training using apex
        dist.init_process_group(backend=args.backend)
        torch.cuda.set_device(args.local_rank)
        print(f'Now local rank is {args.local_rank}')

    # get the device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # retrieve data
    print('Loading the datasets...')
    normalize_info = torch.load(args.data_normalize_info)
    transform = torchvision.transforms.Normalize(normalize_info['mean'], normalize_info['std'])
    train_data, dev_data = utils.LunarDataset(args.train_data, transform=transform), utils.LunarDataset(args.dev_data, transform=transform)
    print('- done.')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                           pin_memory=True, sampler=train_sampler)
    dev_dl = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                         pin_memory=True)

    # define model
    model = UNet(n_channels=1, n_classes=args.num_classes, bilinear=False).to(args.device)

    # define optimizer and scheduler
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if args.num_classes > 1 else 'max', patience=2)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O0')
        if args.distributed:
            model = DDP(model)

    # get the criterion
    criterion = globals()[args.criterion]

    min_loss = 1e8
    for epoch in range(args.epochs):
        train_metric = utils.train_epoch(model, train_dl, optimizer, epoch, criterion, args)
        output_dir = None if epoch % 20 != 0 else args.pred_img_out_dir
        dev_metric = utils.evaluate_model(model, dev_dl, criterion, args, output_dir=output_dir)

        utils.print_metric(args.criterion, train_metric, epoch, mode='Train')
        utils.print_metric(args.criterion, dev_metric, epoch, mode='Test')

        # adjust learning rate
        loss = utils.get_loss_from_metric(args.criterion, dev_metric)
        scheduler.step(loss)

        # update min_loss
        min_loss = min(min_loss, loss)

        print('Min Loss: %.2f' % min_loss)


def main():
    if args.prepare_data:
        prepare_data()

    if args.train:
        train_and_evaluate()

if __name__ == '__main__':
    main()
