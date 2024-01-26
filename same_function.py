import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks.DD import UnFNet_singal
import torch.nn.functional as F
from dataloaders.dataset import TwoStreamBatchSampler
from dataloaders.ISICload import ISIC
from transforms.input import GaussianNoise, EnhanceContrast, EnhanceColor
from transforms.target import Opening, ConvexHull, BoundingBox
import funcy
from skimage.morphology import square
from torch.nn import functional as F
from utils import ramps,dice_score
import ipdb
import torch.nn as nn
from lib.models import nnU_net
import os
# os.environ["WANDB_DISABLED"] = "true"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_bs", type=int, default=2, help='labeled_batch_size')
    parser.add_argument("--max_iterations", type=int, default=15000,
                        help="maxiumn epoch to train")
    ######################################################
    parser.add_argument('--consistency_type', type=str,
                        default="mse", help='consistency_type')
    parser.add_argument('--consistency', type=float,
                        default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=200.0, help='consistency_rampup')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
    ###################################################
    parser.add_argument("--img_size", type=tuple, default=(256, 256), help="fixed size for img&label")
    parser.add_argument("--imgdir", type=str, default='dataset/isic2017/train/images', help="path of img")
    # parser.add_argument("--labeldir", type=str, default='dataset/train/supervised/multi30', help="path of label")
    parser.add_argument("--labeldir", type=str, default='dataset/isic2017/train/masks', help="path of label")
    parser.add_argument("--valdir", type=str, default='dataset/isic2017/val/images', help="path of validation img")
    parser.add_argument("--valsegdir", type=str, default='dataset/isic2017/val/masks', help="path of validation label")
    parser.add_argument('--deterministic', type=int, default=0,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--base_lr', type=float, default=1e-4,
                        help='segmentation network learning rate')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='output channel of network')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='batch_size per gpu')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--label_unlabel', type=str, default='100-1400', help='100-1400,300-1200,500-1000')
    parser.add_argument('--baseline', type=int, default=100, help='100, 1500')
    
    parser.add_argument('--threshold', type=float, default=0.5, help="the threshold of winner (superPixel block)")
    args = parser.parse_args()
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    args.device = device
    return args
# def evaluate_jaccard(outputs, targets):
#     eps = 1e-15
#     intersection = (outputs * targets).sum()
#     union = outputs.sum() + targets.sum()
#     jaccard = (intersection + eps) / (union - intersection + eps)
#     return jaccard
def create_model(args, ema=False):
    model = UnFNet_singal(3, 2, args.device, l_rate=args.base_lr, pretrained=True, has_dropout=ema)
    if ema:
        for param in model.parameters():
            param.detach_()  # TODO:反向传播截断
    return model

def calculate_iou(matrix1, matrix2):
    intersection = np.logical_and(matrix1, matrix2)
    intersection_count = np.count_nonzero(intersection == 1)
    matrix1_count = np.count_nonzero(matrix1 == 1)
    matrix2_count = np.count_nonzero(matrix2 == 1)
    iou = intersection_count / float(matrix1_count + matrix2_count - intersection_count)
    return iou

#data deal
augmentations = [
	GaussianNoise(0, 2),
	EnhanceContrast(0.5, 0.1),
	EnhanceColor(0.5, 0.1)
]
available_conditioning = {
	"original": lambda x: x,
	"opening": Opening(square, 5),
	"convex_hull": funcy.rcompose(Opening(square, 5), ConvexHull()),
	"bounding_box": funcy.rcompose(Opening(square, 5), BoundingBox()),
}
def update_ema_variables(model, ema_model, alpha, global_step):
	# Use the true average until the exponential average is more correct
	alpha = min(1 - 1 / (global_step + 1), alpha)
	for ema_param, param in zip(ema_model.parameters(), model.parameters()):
		ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

#val_epoch
def val_epoch(phase, epoch, model, dataloader,device, experiment, args):
    progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
    val = phase == "val"

    if val:
        model.eval()

    # jacces = []
    dices = []
    ioues = []


    for data in progress_bar:
        volume_batch, label_batch = data["image"], data["mask"]

        volume_batch = volume_batch.to(device, dtype=torch.float32)
        label_batch = label_batch.to(device, dtype=torch.long)
        with torch.no_grad():
            mask_pred = model(volume_batch)
        mask_true = label_batch.squeeze(dim=1)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float().cpu()
        mask_pred_0 = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float().cpu()
        # jacc = evaluate_jaccard(mask_pred_0[:, 1:2, ...], mask_true[:, 1:2, ...])
        dice = dice_score.dice_coeff(mask_pred_0[:, 1:2, ...], mask_true[:, 1:2, ...], reduce_batch_first=False)
        iou = calculate_iou(np.array(mask_pred_0[:, 1:2, ...].to('cpu')),np.array(mask_true[:, 1:2, ...].to('cpu')))

        # jacces.append(jacc)
        dices.append(dice)
        ioues.append(iou)

        progress_bar.set_postfix(dices=np.mean(dices), ioues = np.mean(ioues))

    info = {"dice":round(np.mean(dices), 5),"iou":round(np.mean(ioues), 5)}
    experiment.log({
        'dice': round(np.mean(dices), 5),
        "iou": round(np.mean(ioues), 5)
    })
    if epoch >= 190:
        save_results(path = args.save_name, result = info, args = args)
        if epoch == 199:
            torch.save(model.state_dict(), args.save_last_name)
    return info

def get_data(args):
	labeled = {'100-1400':100,'300-1200':300,'500-1000':500}
	train_preprocess_fn = available_conditioning["original"]
	val_preprocess_fn = available_conditioning['original']
	# 载入数据
	db_train = ISIC(args.imgdir, args.labeldir, size=args.img_size, augmentations=None,
						 target_preprocess=train_preprocess_fn, phase='training')

	db_val = ISIC(args.valdir, args.valsegdir, size=args.img_size, augmentations=None,
					   target_preprocess=val_preprocess_fn, phase='val')
	total_slices = len(db_train)
    
	labeled_idx = list(range(0, labeled[args.label_unlabel]))
	unlabeled_idx = list(set(list(range(0, total_slices))) - set(labeled_idx))

	batch_sampler = TwoStreamBatchSampler(
		labeled_idx, unlabeled_idx, args.batch_size, args.batch_size - args.labeled_bs)
	dataloaders = {}
	dataloaders["train"] = DataLoader(db_train, batch_sampler=batch_sampler,
									  num_workers=0, pin_memory=True)
	dataloaders["validation"] = DataLoader(db_val, batch_size=1, num_workers=0, pin_memory=True, shuffle=False)
	return dataloaders,total_slices

def save_results(path, result, args):
	with open('results_txt/' + path, 'a') as file:
		for key,v in result.items():
			file.write(args.label_unlabel + '_' + ':{},{:.4f} '.format(key, v))
			print(key,v)
		file.write('\n')

def get_current_consistency_weight(args, epoch):
	# Consistency ramp-up from https://arxiv.org/abs/1610.02242
	return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
	# Use the true average until the exponential average is more correct
	alpha = min(1 - 1 / (global_step + 1), alpha)
	for ema_param, param in zip(ema_model.parameters(), model.parameters()):
		ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

# def create_model(args, ema = False):
#     pool_op_kernel_sizes = [
#         [2, 2],
#         [2, 2],
#         [2, 2],
#         [2, 2],
#         [2, 2],
#     ]
#     conv_kernel_sizes = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
#     net_params = {
#         "input_channels": 3,
#         "base_num_features": 32,
#         "num_classes": 2,
#         "num_pool": len(pool_op_kernel_sizes),
#         "num_conv_per_stage": 2,
#         "feat_map_mul_on_downscale": 2,
#         "conv_op": nn.Conv2d,
#         "norm_op": nn.BatchNorm2d,
#         "norm_op_kwargs": {"eps": 1e-5, "affine": True},
#         "dropout_op": nn.Dropout2d,
#         "dropout_op_kwargs": {"p": 0, "inplace": True},
#         "nonlin": nn.LeakyReLU,
#         "nonlin_kwargs": {"negative_slope": 1e-2, "inplace": True},
#         "deep_supervision": False,
#         "dropout_in_localization": False,
#         "final_nonlin": lambda x: x,
#         "pool_op_kernel_sizes": pool_op_kernel_sizes,
#         "conv_kernel_sizes": conv_kernel_sizes,
#         "upscale_logits": False,
#         "convolutional_pooling": True,
#         "convolutional_upsampling": True,
#     }

#     net = nnU_net.Generic_UNet(**net_params)
#     net = net.to(args.device)
    
#     if ema:
#         for param in net.parameters():
#             param.detach_()  # TODO:反向传播截断
#     return net