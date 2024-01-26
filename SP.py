import os
import numpy as np
import torch
import random
from tqdm import tqdm
from networks.DD import UnFNet_singal
from utils.losses import DiceLoss, softmax_mse_loss
import ipdb
from getMask import getMask01
import torch.backends.cudnn as cudnn
import wandb
import os
# os.environ["WANDB_DISABLED"] = "true"

from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn import functional as F
import random
from same_function import get_data,get_args,get_current_consistency_weight,update_ema_variables,val_epoch
#定义一些保存路径
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# ex = Experiment()
fs_observer = os.path.join(BASE_PATH, "results")
if not os.path.exists(fs_observer):
	os.makedirs(fs_observer)
np.set_printoptions(threshold=np.inf)

args = get_args()
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
args.device = device
loss_fn = CrossEntropyLoss()
bceLoss = torch.nn.BCELoss(reduction='none') 
dice_loss = DiceLoss(2)

lu = args.label_unlabel
save_best_name = "Sp_best_" + str(args.label_unlabel) + '.pth'
"wandb initial"
experiment = wandb.init(project='SP',name='test', resume='allow', anonymous='must')

#The MT 
def train_epoch(phase, epoch, model, ema_model, dataloader):
    progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
    training = phase == "train"

    total_losses = []
    iter_num = 0
    l_sup_losses = []
    un_sup_losses =[]
    con_losses = []

    if training:
        model.train()
        ema_model.train()

    for data in progress_bar:
        volume_batch, label_batch, superPixelLabel, name = data["image"], data["mask"], data['superpixel'],data['name']
        volume_batch = volume_batch.to(device, dtype=torch.float32)
        targets = label_batch.to(device, dtype=torch.long)

        labeled_targets = targets[:args.labeled_bs].squeeze(dim=1)

        outputs = model(volume_batch)
        outputs_soft = torch.sigmoid(outputs)
        
        '''add'''
        # _, labeled_targets = getMask01(labeled_targets, superPixelLabel[:args.labeled_bs].squeeze(dim=1),args)
        labeled_sup_loss = torch.mean(loss_fn(outputs_soft[:args.labeled_bs], labeled_targets)) + dice_loss(outputs_soft[:args.labeled_bs], labeled_targets.unsqueeze(1))
        
        consistency_weight = get_current_consistency_weight(args,iter_num // 150)
        if epoch <= 50:
            unlabeled_sup_loss = torch.tensor(0,dtype=float)
            consistency_loss = torch.tensor(0,dtype=float)
        else:
            # '''ema input'''
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise            
            ema_preds = ema_model(ema_inputs)
            ema_preds_soft = torch.sigmoid(ema_preds)
            
            #get the mask and refine preMask
            superPixelLabel = superPixelLabel[args.labeled_bs:].squeeze(dim=1)
            predMask = ema_preds_soft.argmax(dim=1)
            
            Mask01, predMaskRefine = getMask01(predMask, superPixelLabel, args)
            Mask01_expanded = Mask01.clone()
            Mask01_expanded = Mask01_expanded.unsqueeze(1).expand(-1, 2, -1, -1)
            #the loss of supervised loss on unlabeled data
            predMaskRefine_oneHot = F.one_hot(predMaskRefine, 2).permute(0, 3, 1, 2).float()
            
            loss_pixel = bceLoss(outputs_soft[args.labeled_bs:], predMaskRefine_oneHot)
            # ipdb.set_trace()
            
            unlabeled_sup_loss = torch.sum(Mask01_expanded * loss_pixel) / torch.sum(Mask01_expanded + 1e-6) + dice_loss(Mask01_expanded*(outputs_soft[args.labeled_bs:]), (Mask01*predMaskRefine).unsqueeze(1))
            
            # unlabeled_sup_loss = torch.mean(loss_fn(outputs_soft[args.labeled_bs:], predMaskRefine)) #+ dice_loss(Mask01_expanded*(outputs_soft[args.labeled_bs:]), (Mask01*predMask).unsqueeze(1))
            #consistency loss on unlabeled data
            inv_mask = (Mask01_expanded == 0)
            consistency_dist = softmax_mse_loss(outputs[args.labeled_bs:], ema_preds)
            # ipdb.set_trace()
            consistency_loss = torch.sum(inv_mask*consistency_dist)/(torch.sum(inv_mask)+1e-16)
            
        '''wandb post data'''
        experiment.log({
                'labeled_image': wandb.Image(volume_batch[0].cpu()),
                'Pred':wandb.Image(outputs_soft.argmax(dim=1)[0].float().cpu()),
                'GT':wandb.Image(labeled_targets[0].float().cpu()),
                'unlabeled_image':wandb.Image(volume_batch[1].cpu()),
                'pred_mask':wandb.Image(predMask[0].float().cpu()),
                'refine_mask':wandb.Image(predMaskRefine[0].float().cpu()),
                'superpixel mask':wandb.Image(Mask01[0].float().cpu()),
                })

        # #hyper-parameters
        # unlabeled_sup_loss = torch.tensor(0,dtype=float)
        # consistency_loss = torch.tensor(0,dtype=float)
        
        # # labeled_sup_loss *= 0.5
        unlabeled_sup_loss *= consistency_weight
        # consistency_loss *= consistency_weight
        total_loss = labeled_sup_loss + unlabeled_sup_loss + consistency_loss

        model.zero_grad()
        total_loss.backward()
        model.optimize()

        update_ema_variables(model, ema_model, args.ema_decay, iter_num)

        iter_num = iter_num + 1

        l_sup_losses.append(labeled_sup_loss.item())
        un_sup_losses.append(unlabeled_sup_loss.item())
        con_losses.append(consistency_loss.item())
        
        total_losses.append(total_loss.item())

        progress_bar.set_postfix(total_loss=np.mean(total_losses),labeled_sup_loss=np.mean(l_sup_losses),
                                 un_sup_losses=np.mean(un_sup_losses),
                                consistency_loss = np.mean(con_losses))
        
        if iter_num % 2000 == 0:
            model.update_lr()
    experiment.log({
        'total_loss': np.mean(total_losses),
        "labeled_sup_loss": np.mean(l_sup_losses),
        "un_sup_losses": np.mean(un_sup_losses),
        "consistency_loss": np.mean(con_losses),
        "consistency_weight":consistency_weight,
    })
    mean_loss = np.mean(total_losses)
    info = {"loss": mean_loss}
    return info
    
def main(args, device):
    
    base_lr = args.base_lr
    
    def create_model(ema=False):
        # Network definition
        # =======Net========
        # 暂时为False 没有dropout
        model = UnFNet_singal(3, 2, device, l_rate=base_lr, pretrained=True, has_dropout=ema)
        if ema:
            for param in model.parameters():
                param.detach_()  # TODO:反向传播截断
        return model

    model = create_model()  # TODO:创建teacher model
    ema_model = create_model(ema=True)  # TODO:创建student model 初始参数一样

    best_model_path = os.path.join(fs_observer, save_best_name)
    dataloaders, total_slices = get_data(args)

    info = {}
    epochs = range(0, 200)
    best_metric = "dice"
    best_value = 0
    for epoch in epochs:
        info["train"] = train_epoch("train", epoch, model=model,ema_model=ema_model, dataloader=dataloaders["train"])
        info["validation"] = val_epoch("val", epoch, model=model, dataloader=dataloaders["validation"],device=device,experiment=experiment,args=args)
        
        if info["validation"][best_metric] > best_value:
            best_value= info["validation"][best_metric]
            torch.save(model.state_dict(), best_model_path)

    # result = val_epoch("val", epoch, model=model, dataloader=dataloaders["validation"],device=device)#last

    #写入

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

