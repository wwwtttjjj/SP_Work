import os
import numpy as np
import torch
import random
from tqdm import tqdm
from utils.losses import DiceLoss
import ipdb
import torch.backends.cudnn as cudnn
import wandb

from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn import functional as F
import random
from same_function import create_model,get_data,get_args,get_current_consistency_weight,update_ema_variables,val_epoch
#定义一些保存路径
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
fs_observer = os.path.join(BASE_PATH, "results")
if not os.path.exists(fs_observer):
	os.makedirs(fs_observer)
np.set_printoptions(threshold=np.inf)

args = get_args()

loss_fn = CrossEntropyLoss()
bceLoss = torch.nn.BCELoss(reduction='none') 
dice_loss = DiceLoss(2)

save_name = "UAMT" + str(args.label_unlabel) + '.txt'
save_best_name = "UAMT_best_" + str(args.label_unlabel) + '.pth'
save_last_name = "UAMT_last_" + str(args.label_unlabel) + '.pth'

args.save_name = save_name
args.save_best_name = save_best_name
args.save_last_name = save_last_name

def sharpening(P):
	T = 1 / args.temperature
	P_sharpen = P ** T / (P ** T + (1 - P) ** T)
	return P_sharpen
"wandb initial"
experiment = wandb.init(project='SP',name='UAMT'+str(args.label_unlabel), resume='allow', anonymous='must')

#The UAMT 
def train_epoch(phase, epoch, model, ema_model, dataloader):
    progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
    training = phase == "train"

    total_losses = []
    iter_num = 0
    l_sup_losses = []
    con_losses = []

    if training:
        model.train()
        ema_model.train()

    for data in progress_bar:
        volume_batch, label_batch = data["image"], data["mask"]
        volume_batch = volume_batch.to(args.device, dtype=torch.float32)
        targets = label_batch.to(args.device, dtype=torch.long)

        labeled_targets = targets[:args.labeled_bs].squeeze(dim=1)

        outputs = model(volume_batch)
        outputs_soft = torch.sigmoid(outputs)
        
        '''add'''
        # _, labeled_targets = getMask01(labeled_targets, superPixelLabel[:args.labeled_bs].squeeze(dim=1),args)
        labeled_sup_loss = torch.mean(loss_fn(outputs_soft[:args.labeled_bs], labeled_targets)) + dice_loss(outputs_soft[:args.labeled_bs], labeled_targets.unsqueeze(1))
        
        consistency_weight = get_current_consistency_weight(args,iter_num // 150)
        if epoch <= args.epoch_unlabeled:
            consistency_loss = torch.tensor(0,dtype=float)
        else:
            # '''ema input'''
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            
            B = args.batch_size - args.labeled_bs
            K = 4
            ema_preds = torch.zeros([K, B, 2, 256, 256]).to(args.device)
            ema_preds_soft = torch.zeros([K, B, 2, 256, 256]).to(args.device)
            
            for k in range(K):
                with torch.no_grad():
                    ema_preds[k]= ema_model(ema_inputs)
                    
            ema_preds_soft = torch.sigmoid(ema_preds)
            ema_uncer =  -1.0 * torch.sum(ema_preds_soft * torch.log2(ema_preds_soft + 1e-6), dim=0)
            ema_preds_soft = sharpening(ema_preds_soft) + ema_uncer
            consistency_loss = torch.mean(
				(outputs_soft[args.labeled_bs:] - ema_preds_soft) ** 2)
            consistency_loss *= consistency_weight
            
        total_loss = labeled_sup_loss + consistency_loss
        model.zero_grad()
        total_loss.backward()
        model.optimize()

        update_ema_variables(model, ema_model, args.ema_decay, iter_num)

        iter_num = iter_num + 1

        l_sup_losses.append(labeled_sup_loss.item())
        con_losses.append(consistency_loss.item())
        total_losses.append(total_loss.item())

        progress_bar.set_postfix(total_loss=np.mean(total_losses),labeled_sup_loss=np.mean(l_sup_losses),
                                consistency_loss = np.mean(con_losses))
        if iter_num % 2000 == 0:
            model.update_lr()
    mean_loss = np.mean(total_losses)
    info = {"loss": mean_loss}
    return info
def main(args):
    model = create_model(args)  # TODO:创建teacher model
    ema_model = create_model(args,ema=True)  # TODO:创建student model 初始参数一样

    best_model_path = os.path.join(fs_observer, save_best_name)
    dataloaders, _ = get_data(args)

    info = {}
    epochs = range(0, 200)
    best_metric = "dice"
    best_value = 0
    for epoch in epochs:
        info["train"] = train_epoch("train", epoch, model=model,ema_model=ema_model, dataloader=dataloaders["train"])
        info["validation"] = val_epoch("val", epoch, model=model, dataloader=dataloaders["validation"],device=args.device,experiment=experiment,args=args)
        
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
	main(args)

