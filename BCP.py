import os
import numpy as np
import torch
import random
from tqdm import tqdm
from utils.losses import DiceLoss
import ipdb
import torch.backends.cudnn as cudnn
import wandb
from skimage.measure import label
import torch.nn as nn

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
args.batch_size = 4 #labeled和unlabeled必须一致

loss_fn = CrossEntropyLoss()
bceLoss = torch.nn.BCELoss(reduction='none') 
dice_loss = DiceLoss(2)

save_name = "BCP" + str(args.label_unlabel) + '.txt'
save_best_name = "BCP_best_" + str(args.label_unlabel) + '.pth'
save_last_name = "BCP_last_" + str(args.label_unlabel) + '.pth'

args.save_name = save_name
args.save_best_name = save_best_name
args.save_last_name = save_last_name


"wandb initial"
experiment = wandb.init(project='SP',name='BCP'+str(args.label_unlabel), resume='allow', anonymous='must')
def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i] #== c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs
def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss()
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    # print(output.shape, img_l.unsqueeze(1).shape)
    loss_ce = image_weight * (CE(output, img_l.squeeze(1)) * mask.squeeze(1)).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l.squeeze(1)) * patch_mask.squeeze(1)).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    return loss_ce
sub_bs = int(args.labeled_bs/2)
#The BCP 
def train_epoch(phase, epoch, model, ema_model, dataloader):
    progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
    training = phase == "train"

    total_losses = []
    iter_num = 0


    if training:
        model.train()
        ema_model.train()

    for data in progress_bar:
        volume_batch, label_batch = data["image"], data["mask"]
        volume_batch = volume_batch.to(args.device, dtype=torch.float32)
        targets = label_batch.to(args.device, dtype=torch.long)
        
        if epoch <= args.epoch_unlabeled:
            outputs = model(volume_batch)
            outputs_soft = torch.sigmoid(outputs)
            loss = torch.mean(loss_fn(outputs_soft[:args.labeled_bs], targets[:args.labeled_bs].squeeze(dim=1))) + dice_loss(outputs_soft[:args.labeled_bs], targets[:args.labeled_bs].squeeze(dim=1).unsqueeze(1))
        else:
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs+sub_bs], volume_batch[args.labeled_bs+sub_bs:]
            
            lab_a, lab_b = targets[:sub_bs], targets[sub_bs:args.labeled_bs]
            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)
                img_mask, loss_mask = generate_mask(img_a)

                
            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)
            out_unl = model(net_input_unl)
            out_l = model(net_input_l)
            unl_ce = mix_loss(out_unl, plab_a, lab_a, loss_mask, u_weight=0.5, unlab=True)
            l_ce = mix_loss(out_l, lab_b, plab_b, loss_mask, u_weight=0.5)
            loss_ce = unl_ce + l_ce 
            loss = (loss_ce) / 2
            
        total_loss = loss
        model.zero_grad()
        total_loss.backward()
        model.optimize()

        update_ema_variables(model, ema_model, args.ema_decay, iter_num)

        iter_num = iter_num + 1

        total_losses.append(total_loss.item())

        progress_bar.set_postfix(total_loss=np.mean(total_losses))
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

