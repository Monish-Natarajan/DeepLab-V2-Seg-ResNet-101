import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

import data.transforms_seg as Trs
from configs.defaults import _C
from data.voc import VOC_seg

from models.PolyScheduler import PolynomialLR
from models.SegNet import DeepLab_ASPP, DeepLab_LargeFOV
from utils.densecrf import dense_crf #
from utils.evaluate import evaluate  #
#from utils.wandb import init_wandb, wandb_log_seg

def train(cfg, train_loader, model, checkpoint):
    """
    Training function
    
    Train the model using the training dataloader and the last saved checkpoint.
    Inputs:
    - cfg: config file
    - train_loader: training dataloader
    - model: model
    - checkpoint: state dict of the model, optimizer, scheduler, etc.
    Outputs:
    - Trained model saved locally and on wandb
    """    
    model = model.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=255)
        
    lr = cfg.SOLVER.LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    
    # Creating optimizer for the both models

    optimizer = optim.SGD(
    params=[
        {
            "params":model.get_1x_lr_params(),
            "lr":lr,
            "weight_decay":wd
        },
        {
            "params":model.get_10x_lr_params(),
            "lr":10*lr,
            "weight_decay":wd
        }
    ],
    lr=lr,
    weight_decay=wd,
    momentum=cfg.SOLVER.MOMENTUM
    )

    # Poly learning rate scheduler according to the paper 
    scheduler = PolynomialLR(optimizer, 
                             step_size=cfg.SOLVER.STEP_SIZE, 
                             iter_max=cfg.SOLVER.MAX_ITER, 
                             power=cfg.SOLVER.GAMMA)
    
    ###############################################################
    
    curr_it = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        scheduler.load_state_dict(checkpoint['sched_state_dict'])
        curr_it = checkpoint['iter']

    iterator = iter(train_loader)

    for it in tqdm(range(curr_it+1, cfg.SOLVER.MAX_ITER+1)):
        try:
            sample = next(iterator)
        except:
            iterator = iter(train_loader)
            sample = next(iterator)
        img, masks = sample # VOC_seg dataloader returns image and the corresponing (pseudo) label
        ygt, ypred = masks
        
        model.train()

        # Forward pass
        img = img.to('cuda')
        img_size = img.size()
        logit, feature_map = model(img, (img_size[2], img_size[3]))

        # Loss calculation
        ypred = ypred.cuda().long()
        loss = criterion(logit, ypred)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Update the learning rate using poly scheduler
        scheduler.step()

        train_loss = loss.item()
        # Logging Loss and LR on wandb
        wandb_log_seg(train_loss, optimizer.param_groups[0]["lr"], it)

        save_dir = "./ckpts/"
        if it%1000 == 0 or it == cfg.SOLVER.MAX_ITER:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'sched_state_dict': scheduler.state_dict(),
                'iter': it,
            }
            torch.save(checkpoint, save_dir + str(it) + '.pth')
            if cfg.WANDB.MODE: 
                wandb.save(save_dir + str(it) + '.pth')
            # Evaluate the train_loader at this checkpoint and Log the metrics on WandB
            accuracy, iou = evaluate(cfg, train_loader, model)
            if cfg.WANDB.MODE: 
                # Log on WandB
                wandb.log({
                    "Mean IoU": iou,
                    "Mean Accuracy": accuracy
                    })


def val(cfg, data_loader, model, checkpoint):
    """
    Validation function
    Evaluate the model using the validation dataloader and the last saved checkpoint.
    And apply the CRF post-processing to the predicted masks.
    Inputs:
    - cfg: config file
    - data_loader: validation dataloader
    - model: model
    - checkpoint: state dict of the model, optimizer, scheduler, etc.
    Outputs:
    - Validation metrics saved locally and on wandb
    """
    model = model.cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    accuracy, iou = evaluate(cfg, data_loader, model)
    print("Validation Mean Accuracy ", accuracy)
    print("Validation Mean IoU ", iou)
    wandb.run.summary["Validation Mean Accuracy"] = accuracy
    wandb.run.summary["Validation Mean IoU"] = iou
    # Evaluating the validation dataloader after CRF post-processing
    crf_accuracy, crf_iou = dense_crf(cfg, data_loader, model)
    wandb.run.summary["CRF Validation Mean Accuracy"] = crf_accuracy
    wandb.run.summary["CRF Validation Mean IoU"] = crf_iou   
    print("CRF Validation Mean Accuracy ", crf_accuracy)
    print("CRF Validation Mean IoU ", crf_iou)

   
def main(cfg):
    """
    Main function
    Call the train and val functions according to the mode.
    Inputs:
    - cfg: config file
    Outputs:
    """
    if cfg.SEED:
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED)
        
    if cfg.WANDB.MODE: 
        init_wandb(cfg)
        
    if cfg.DATA.MODE == "train_weak":
        tr_transforms = Trs.Compose([
            Trs.RandomScale(0.5, 1.5),
            Trs.ResizeRandomCrop(cfg.DATA.CROP_SIZE), 
            Trs.RandomHFlip(0.5), 
            Trs.ColorJitter(0.5,0.5,0.5,0),
            Trs.Normalize_Caffe(),
        ])
    elif cfg.DATA.MODE == "val":
        tr_transforms = Trs.Compose([
            Trs.Normalize_Caffe(),
        ])
    else:
        print("Incorrect Mode provided!")
        return

    dataset = VOC_seg(cfg, tr_transforms)
    if cfg.DATA.MODE == "train_weak":
        data_loader = DataLoader(dataset, 
                                 batch_size=cfg.DATA.BATCH_SIZE, 
                                 shuffle=True, 
                                 num_workers=4, 
                                 pin_memory=True, 
                                 drop_last=True)
        
    elif cfg.DATA.MODE == "val":
        data_loader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=4, 
                                 pin_memory=True, 
                                 drop_last=True)
    

    model = DeepLab_ASPP(cfg.DATA.NUM_CLASSES,
                         output_stride=None, 
                         sync_bn=False, 
                         is_CS=True)
     
    # Load pre-trained backbone weights
    state_dict = torch.load(f"./weights/{cfg.MODEL.WEIGHTS}")
    # Manually matching the sate dicts if model and pretrained weights (only for res101)

    for key in list(state_dict.keys()):
        state_dict[key.replace('base.', '')] = state_dict.pop(key)
    
    model.backbone.load_state_dict(state_dict, strict=False)

    # Save model locally and then on wandb
    save_dir = './ckpts/'
    # Load pretrained model from wandb if present
    if cfg.WANDB.CHECKPOINT:
        wandb_checkpoint = wandb.restore('ckpts/' + cfg.WANDB.CHECKPOINT)    
        checkpoint = torch.load(wandb_checkpoint.name)
        print("WandB checkpoint Loaded with iteration: ", checkpoint['iter'])
    else:
        print("WandB checkpoint not Loaded")
        checkpoint = None
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    # Call the appropriate mode from main()
    if cfg.DATA.MODE == "train_weak":
        train(cfg, data_loader, model, checkpoint)
    elif cfg.DATA.MODE == "val":
        val(cfg, data_loader, model, checkpoint)


def get_args():
    """
    Get the arguments from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--gpu-id", type=str, default="0", help="select a GPU index")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    main(cfg)