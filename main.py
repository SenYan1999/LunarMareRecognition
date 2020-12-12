import torch
import utils

from model import UNet
from args import args
from utils.loss_fn import *

def train_and_evaluate():
    # get the device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # retrieve data
    ### TODO: DEFINE TRANSPOSE
    dataset = utils.LunarDataset(args.data)

    train_data_length = int(len(dataset) * args.train_data_ratio)
    train_data, dev_data = torch.utils.data.random_split(dataset, [train_data_length, len(dataset) - train_data_length])

    train_dl = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size)
    dev_dl = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size)

    # define model
    model = UNet(n_channels=1, n_classes=1, bilinear=False).to(args.device)

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # get the criterion
    criterion = globals()[args.criterion]

    for epoch in range(args.epochs):
        train_metric = utils.train_epoch(model, train_dl, optimizer, epoch, criterion, None, args)
        dev_metric = utils.evaluate_model(model, dev_dl, criterion, args, output_dir=args.pred_img_out_dir)

        utils.print_metric(args.criterion, train_metric, epoch, mode='Train')
        utils.print_metric(args.criterion, dev_metric, epoch, mode='Test')

def main():
    if args.prepare_data:
        utils.prepare_data(args.raw_data_dir, args.data)

    if args.train:
        train_and_evaluate()

if __name__ == '__main__':
    main()
