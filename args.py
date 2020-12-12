import argparse

parser = argparse.ArgumentParser()

# multiprocessing and fp16 precision
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--backend', type=str, default='nccl')
parser.add_argument('--fp16', action='store_true')

# train mode
parser.add_argument('--prepare_data', action='store_true')
parser.add_argument('--train', action='store_true')

# data prepare and dataset
parser.add_argument('--raw_data_dir', type=str, default='data/all_segments')
parser.add_argument('--data', type=str, default='data/data.h5')
parser.add_argument('--train_data', type=str, default='data/train_data.h5')
parser.add_argument('--dev_data', type=str, default='data/dev_data.h5')
parser.add_argument('--data_normalize_info', type=str, default='data/normalize_info.pt')
parser.add_argument('--train_data_ratio', type=float, default=0.8)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=4)

# model details
parser.add_argument('--num_classes', type=int, default=1)

# training parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-8)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--criterion', type=str, default='bce_dice_weight_loss')

# evaluate detail
parser.add_argument('--pred_img_out_dir', type=str, default='output/img_pred')

# log details
parser.add_argument('--save_model', type=str, default='output/models/model.pt')

args = parser.parse_args()
