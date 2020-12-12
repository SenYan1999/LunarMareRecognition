import argparse

parser = argparse.ArgumentParser()

# multiprocessing and fp16 precision
parser.add_argument('--fp16', action='store_true')

# train mode
parser.add_argument('--prepare_data', action='store_true')
parser.add_argument('--train', action='store_true')

# data prepare and dataset
parser.add_argument('--raw_data_dir', type=str, default='data/all_segments')
parser.add_argument('--data', type=str, default='data/data.h5')
parser.add_argument('--train_data_ratio', type=float, default=0.8)
parser.add_argument('--batch_size', type=int, default=8)

# model details
parser.add_argument('--num_classes', type=int, default=1)

# training parameters
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--criterion', type=str, default='bce_dice_weight_loss')

# evaluate detail
parser.add_argument('--pred_img_out_dir', type=str, default='output/img_pred')

args = parser.parse_args()
