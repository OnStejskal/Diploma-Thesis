from copy import deepcopy

from torch import device, logical_and, logical_or, no_grad, Tensor, mean
from torch import device, set_grad_enabled
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from SEGan.loss import dice_loss, dice_loss_multiclass
from SEGan.net import NetC, NetS
from common.metrics import classes_IoU, classses_dice_score
import numpy as np
import matplotlib.pyplot as plt
from common.dataset import SegmentationDataset
from common.visualization import plot_image, plot_segmentation
from common.result_saver import ResultSaver

def segan_run_segmentation(
    DEVICE,
    TRAIN_IMG_PATH: str,
    TRAIN_LABELS_PATH: str,
    VAL_IMG_PATH: str,
    VAL_LABELS_PATH: str,
    TRAIN_TRANSFORMATIONS_SEG,
    VAL_TRANSFORMATIONS_SEG,
    TRANSFORMATIONS_TORCH,
    PLAQUE_WITH_WALL = False,
    CATEGORIES = 4,
    EPOCHS = 1
):
    train_dataset = SegmentationDataset(
        TRAIN_IMG_PATH, TRAIN_LABELS_PATH, TRAIN_TRANSFORMATIONS_SEG, TRANSFORMATIONS_TORCH, plaque_with_wall=PLAQUE_WITH_WALL
    )

    val_dataset = SegmentationDataset(
        VAL_IMG_PATH, VAL_LABELS_PATH, VAL_TRANSFORMATIONS_SEG, TRANSFORMATIONS_TORCH, plaque_with_wall=PLAQUE_WITH_WALL
    )
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    number_of_classes_without_background = CATEGORIES - 1
    lr = 0.00002  # Learning Rate. Default=0.0002
    beta1 = 0.5  # beta1 for adam
    decay = 0.5  # Learning rate decay
    alpha = 1 #0.1  # weight given to dice loss while generator training
    

    return train_models(
        train_loader,
        val_loader,
        DEVICE,
        EPOCHS,
        number_of_classes_without_background,
        alpha,
        beta1,
        decay,
        lr
    )

def train_mode(models):
    for model in models:
        model.train()

def eval_mode(models):
    for model in models:
        model.eval()

def train_models(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: device,
    num_epochs: int,
    n_classes: int,
    alpha: float,
    beta1 = float,
    decay = float,
    lr = float

    
) -> list[Module]:

    train_size = len(train_loader)
    val_size = len(val_loader)
    print(f'train size: {train_size}, val size: {val_size}')

    g_model = NetS(classes=n_classes+1)
    g_model.to(device)
    g_optimizer = RMSprop(g_model.parameters(), lr=lr)
    # g_optimizer = RMSprop(g_model.parameters(), lr=lr, )

    d_models = []
    d_optimizers = []
    for i in range(n_classes):
        d_model = NetC()
        d_model.to(device)
        d_optimizer = RMSprop(d_model.parameters(), lr=lr)
        d_models.append(d_model)
        d_optimizers.append(d_optimizer)


    # best_val_loss = 10 ** 8
    best_model = None
    max_iou = 0

    train_disc_l1_loss = []
    mean_train_disc_l1_loss = []
    train_gen_l1_loss = []
    train_gen_dice_loss = []
    train_gen_joint_loss = []

    val_joint_loss = []
    val_l1_loss = []
    val_dice_loss = []
    val_dice_score = []
    val_iou = []
    mean_val_dice_score = []
    mean_val_iou = []

    rs = ResultSaver()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 12)

        epoch_disc_l1_loss = [0 for _ in range(n_classes)]
        epoch_gen_l1_loss = 0
        epoch_gen_dice_loss = 0
        epoch_gen_joint_loss = 0
        
        g_model.train()
        for image, target_all_class in train_loader:
            image = image.to(device) # [batch x 3 x 512 x 512]
            target_all_class = target_all_class.to(device) # [batch x 512 X 512]


            ##################################
            ### train Discriminator/Critic ###
            ##################################
            output_all_class = g_model(image)
            output_all_class = F.softmax(output_all_class, dim=1) # [batch, 4, w,h]
            output_all_class = output_all_class.detach() ### detach G from the network
            # plot_segmentation(output_all_class[0].argmax(dim = 0))
            for i in range(n_classes):
                target = (target_all_class == i + 1).to(torch.int)
                target = target.to(device) # [batch x 512 X 512]
                output = output_all_class[:,i+1,:,:].unsqueeze(1)
                d_models[i].zero_grad()

                # in case we do not do any argmax we just take the probabiluty of output of class i and multiply it the original image
                
                output_masked = image * output.expand_as(image)
                # print(output_masked.shape)
                # plot_image(output_masked[0])
                output_masked = output_masked.to(device)

                target_masked = image * target.unsqueeze(1).expand_as(image)
                # plot_image(target_masked[0])
                target_masked = target_masked.to(device)

                output_D = d_models[i](output_masked)
                target_D = d_models[i](target_masked)
                loss_D = - torch.mean(torch.abs(output_D - target_D))
                loss_D.backward()
                d_optimizers[i].step()
                epoch_disc_l1_loss[i] += loss_D.item()
                ## clip parameters in D
                for p in d_models[i].parameters():
                    p.data.clamp_(-0.05, 0.05)

            # print(f'discl1 loss: {epoch_disc_l1_loss}')
            #################################
            ### train Generator/Segmentor ###
            #################################
            g_model.zero_grad()

            output_all_class = g_model(image)
            output_all_class = F.softmax(output_all_class, dim = 1)

            loss_dice = dice_loss(output_all_class,target_all_class, include_background=True)
            
            losses_G = []
            for i in range(n_classes):
                target = (target_all_class == i + 1).to(torch.int)
                target = target.to(device) # [batch x 512 X 512]
                output = output_all_class[:,i+1,:,:].unsqueeze(1)
                output_masked = image * output.expand_as(image)
                output_masked = output_masked.to(device)
                target_masked = image * target.unsqueeze(1).expand_as(image)
                target_masked = target_masked.to(device)
                output_G = d_models[i](output_masked)
                target_G = d_models[i](target_masked)
                losses_G.append(torch.mean(torch.abs(output_G - target_G)))
            loss_G = sum(losses_G)/len(losses_G)
            loss_G_joint = loss_G + alpha * loss_dice
            # loss_G_joint = loss_dice
            loss_G_joint.backward()
            g_optimizer.step()
            epoch_gen_joint_loss += loss_G_joint.item()
            epoch_gen_dice_loss += loss_dice.item()
            epoch_gen_l1_loss += loss_G_joint.item()

        # for i in range(n_classes):
            # train_disc_l1_loss[i].append(epoch_disc_l1_loss[i]/train_size)
        epoch_disc_l1_loss = [x/train_size for x in epoch_disc_l1_loss]
        train_disc_l1_loss.append(epoch_disc_l1_loss)
        mean_train_disc_l1_loss.append(sum(epoch_disc_l1_loss)/n_classes)
        train_gen_l1_loss.append(epoch_gen_l1_loss/train_size)
        train_gen_dice_loss.append(epoch_gen_dice_loss/train_size)
        train_gen_joint_loss.append(epoch_gen_joint_loss/train_size)

        ##################################
        ## validate Generator/Segmentor ##
        ##################################
        g_model.eval()
        d_model.eval()
        epoch_iou = [0 for i in range(n_classes + 1)]
        epoch_dice_score = [0 for i in range(n_classes + 1)]
        epoch_val_joint_los = 0
        epoch_val_dice_loss = 0
        epoch_val_l1_loss = 0
        with set_grad_enabled(False):
            for img, target_all_class in val_loader:
                img = img.to(device)
                target_all_class = target_all_class.to(device)
                pred_all = g_model(img)
                pred_all = F.softmax(pred_all)
                loss_dice = dice_loss(pred_all, target_all_class)
                iou = classes_IoU(pred_all, target_all_class)
                dice_score = classses_dice_score(pred_all, target_all_class)
                epoch_val_dice_loss += loss_dice.item()
                for i in range(len(iou)):
                    epoch_iou[i] += iou[i]
                    epoch_dice_score[i] += dice_score[i]
        g_model.train()

        val_dice_loss.append(epoch_val_dice_loss/val_size)
        for i in range(n_classes + 1):
            val_iou[i].append(epoch_iou[i]/val_size)
            val_dice_score[i].append(epoch_dice_score[i]/val_size)
        mean_val_iou.append(sum(val_iou[i][-1] for i in range(n_classes+1))/(n_classes+1))
        mean_val_dice_score.append(sum(val_dice_score[i][-1] for i in range(n_classes+1))/(n_classes+1))
            
        for i in range(n_classes+1):
            print(f"--------- Class {i} ---------")
            print(f'IoU class {i}: {val_iou[i][-1]:.4f}')
            print(f'Dice class {i}: {val_dice_score[i][-1]:.4f}')
        
        for i in range(n_classes):
            print(f"--------- Disc {i+1} ---------")
            print(f'Train Discriminator loss: {train_disc_l1_loss[i][-1]:.4f}')
        
        print(f"--------- Mean Score ---------")
        print(f'Train generator loss: {train_gen_joint_loss[-1]:.4f}')
        print(f'Train dice loss: {train_gen_dice_loss[-1]:.4f}')
        print(f'Train mean l1 generator loss: {train_gen_l1_loss[-1]:.4f}')
        print(f'Train mean l1 discriminator loss: {mean_train_disc_l1_loss[-1]:.4f}')
        print(f'Val dice loss: {val_dice_loss[-1]:.4f}')
        print(f'mean IoU: {mean_val_iou[-1]:.4f}')
        print(f'mean Dice score: {mean_val_dice_score[-1]:.4f}')

        mIoU = mean_val_iou[-1]
        if mIoU > max_iou:
            max_iou = mIoU
            best_model = deepcopy(g_model.state_dict())
    
        if epoch % 25 == 0:
            lr = lr*decay
            if lr <= 0.00000001:
                lr = 0.00000001
            print('Learning Rate: {:.6f}'.format(lr))
            # print('K: {:.4f}'.format(k))
            print('Max mIoU: {:.4f}'.format(max_iou))
            g_optimizer = RMSprop(g_model.parameters(), lr=lr)
            for i in range(n_classes):
                d_optimizers[i] = RMSprop(d_models[i].parameters(), lr=lr)

        print('-------------------------------------------------------------------------------------------------------------------')
        print()

        g_model.load_state_dict(best_model)
    result_dict = {
        'train_disc_l1_loss': train_disc_l1_loss,
        'mean_train_disc_l1_loss': mean_train_disc_l1_loss,
        'train_gen_l1_loss': train_gen_l1_loss,
        'train_gen_dice_loss': train_gen_dice_loss,
        'train_losses': train_gen_joint_loss,
        'val_joint_loss': val_joint_loss,
        'val_l1_loss': val_l1_loss,
        'val_losses': val_dice_loss,
        'val_dices_classes': val_dice_score,
        'val_ious_classes': val_iou,
        'val_dices_mean': mean_val_dice_score,
        'val_ious_mean': mean_val_iou,
    }
    return g_model, result_dict


    
