import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from tqdm import tqdm

from dataloaders.ISICload import ISIC
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import CrossEntropyLoss
from same_function import get_args,val_epoch,available_conditioning,create_model
import ipdb
import wandb
from utils.losses import DiceLoss
from torch.utils.data import DataLoader, SubsetRandomSampler

dice_loss = DiceLoss(2)
"wandb initial"
experiment = wandb.init(project='SP',name='LinKNet_test', resume='allow', anonymous='must')
BASE_PATH = os.path.dirname(os.path.abspath(__file__))  
fs_observer = os.path.join(BASE_PATH, "results")
if not os.path.exists(fs_observer):
	os.makedirs(fs_observer)
np.set_printoptions(threshold=np.inf)

args = get_args()
save_name = "LinkNet" + str(args.baseline) + ".txt"
save_best_name = "LinkNet_best_" + str(args.baseline) + '.pth'
save_last_name = "LinkNet_last_" + str(args.baseline) + '.pth'

args.save_name = save_name
args.save_best_name = save_best_name
args.save_last_name = save_last_name

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def get_data(args):
    train_preprocess_fn = available_conditioning["original"]
    val_preprocess_fn = available_conditioning['original']
    # 载入数据
    db_train = ISIC(args.imgdir, args.labeldir, size=args.img_size, augmentations=None,
                            target_preprocess=train_preprocess_fn, phase='training')

    db_val = ISIC(args.valdir, args.valsegdir, size=args.img_size, augmentations=None,
                        target_preprocess=val_preprocess_fn, phase='val')
    
    total_slices = len(db_train)
    subset_indices = list(range(args.baseline))
    sampler = SubsetRandomSampler(subset_indices)

    dataloaders = {
        "train": DataLoader(db_train, batch_size=args.labeled_bs, num_workers=0, pin_memory=True, sampler=sampler),
        "validation": DataLoader(db_val, batch_size=1, num_workers=0, pin_memory=True, shuffle=False)
    }

    return dataloaders,total_slices

#train_epoch
def train_epoch(phase, epoch, model, dataloader, loss_fn):
    progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
    training = phase == "train"
    losses_unet = []
    iter_num = 0

    if training:
        model.train()

    for data in progress_bar:
        volume_batch, label_batch = data["image"], data["mask"]
        volume_batch = volume_batch.to(device, dtype=torch.float32)
        targets = label_batch.to(device, dtype=torch.long)

        targets = targets[:args.labeled_bs].squeeze(dim=1)

        outputs = model(volume_batch)
        outputs_soft = torch.sigmoid(outputs)
        loss = torch.mean(loss_fn(outputs_soft[:args.labeled_bs], targets)) + dice_loss(outputs_soft[:args.labeled_bs], targets.unsqueeze(1))

        model.zero_grad()
        loss.backward()
        model.optimize()

        iter_num = iter_num + 1
        losses_unet.append(loss.item())

        progress_bar.set_postfix(loss_unet=np.mean(losses_unet))
        if iter_num % 2000 == 0:
            model.update_lr()

    mean_loss = np.mean(losses_unet)
    info = {"loss": mean_loss,
        }
    return info

def main(args, device):
    model = create_model(args)

    best_model_path = os.path.join(fs_observer, save_best_name)

    dataloaders,_ = get_data(args)
    loss_fn = CrossEntropyLoss(reduce='none')
    info = {}
    epochs = range(0, 200)
    best_metric = "dice"
    best_value = 0


    for epoch in epochs:
        info["train"] = train_epoch("train", epoch, model=model, dataloader=dataloaders["train"], loss_fn=loss_fn)
        info["validation"] = val_epoch("val", epoch, model=model, dataloader=dataloaders["validation"],device=device,experiment=experiment,args=args)
        
        if info["validation"][best_metric] > best_value:
            best_value= info["validation"][best_metric]
            torch.save(model.state_dict(), best_model_path)

if __name__ == '__main__':
	if not args.deterministic:
		cudnn.benchmark = True
		cudnn.deterministic = False
	else:
		cudnn.benchmark = False
		cudnn.deterministic = True

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	main(args, device)