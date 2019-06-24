from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
from torch.utils.data import Dataset, DataLoader
import torchvision
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#udaRuntimeGetVersion()
plt.ion()   # interactive mode

import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import imgaug as ia
from tensorboardX import SummaryWriter


from confusion_matrix_plot import plot_confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from data_load2 import make_dataset
from new_densenet import NewDensenet


##view some images
def imshow(inp, title, fig):
    """imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(fig, figsize=(10, 10))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(2)

def image_show(dataset_train, dataset_val):
    ##Show train batch images
    dataloader_ = DataLoader(dataset_train, batch_size=6, shuffle=True, num_workers=8)
    inputs, classes = next(iter(dataloader_))
    classes = [item.item() for item in classes]
    out = torchvision.utils.make_grid(inputs, padding=100)

    imshow(out, classes, 1)  # title = [ class_names[x] for x in classes])

    plt.figure(2, figsize=(7, 7))
    plt.imshow(inputs[0].numpy().transpose(1, 2, 0))
    plt.title('1st image zoom in')
    plt.pause(1)

    ##Show val batch images
    dataloader_val = DataLoader(dataset_val, batch_size=6, shuffle=True, num_workers=8)
    inputs, classes = next(iter(dataloader_val))
    classes = 'val_dataset'
    out = torchvision.utils.make_grid(inputs, padding=100)

    imshow(out, classes, 3)

    # plt.close(fig='all')

def main():
    writer = SummaryWriter(max_queue=10000)
    writer_logdir = str(writer.log_dir)[5:]
    print('writer_logdir :', writer_logdir)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    ia.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
    torch.cuda.manual_seed_all(100)

    Model = writer_logdir
    if not os.path.isdir('./{}'.format(Model)):
        os.mkdir('./{}'.format(Model))


    ##for matplotlib absolute colorbar
    levels = [0, 1, 2, 3, 4]
    colors = ['green', 'yellow', 'blue', 'red']
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)


    device = 'cuda'

    nets = [NewDensenet() for _ in range(3)]  # 3 is K-fold number
    optimizers = [optim.Adam(nets[i].parameters(), lr=0.001, weight_decay=0.00001) for i in range(3)]
    schedulers = [optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40, 50], gamma=0.3) for optimizer in
                  optimizers]
    criterion = nn.NLLLoss(weight=None, reduction='mean')
    dataset = make_dataset() #[(dataset_train0,dataset_val0),(dataset_train1,dataset_val),(dataset_train1,dataset_val1)]

    ##viewing some images for sanity check
    #image_show(dataset_train=dataset[0][0], dataset_val=dataset[0][1])

    for k_fold in [2, 1, 0]:
        Dataloader_train = DataLoader(dataset[k_fold][0], batch_size=2, shuffle=True, num_workers=4, drop_last=True)
        Dataloader_val = DataLoader(dataset[k_fold][1], batch_size=2, shuffle=True, num_workers=4, drop_last=True)

        total_iter_train = [40 * len(Dataloader_train), 40 * len(Dataloader_train), 40 * len(Dataloader_train)]
        total_iter_val = [40 * len(Dataloader_val), 40 * len(Dataloader_val), 40 * len(Dataloader_val)]

        net = nets[k_fold]
        net = nn.DataParallel(net)
        net.to(device)

        optimizer = optimizers[k_fold]
        scheduler = schedulers[k_fold]

        running_loss = 0
        iter_train = 0
        iter_val = 0

        train_acc_whole_epoch = [0, ]
        val_acc_whole_epoch = [0, ]
        train_acc_one_epoch = 0
        val_acc_one_epoch = 0

        best_epoch = None
        best_model_st_dct = None
        best_optimizer_st_dct = None
        best_val_acc = 0

        epoch = 0
        whole_epoch = 40
        while epoch < whole_epoch:
            scheduler.step()
            for i, (images, targets) in enumerate(Dataloader_train):
                iter_train += 1
                net.train()

                images = images.type('torch.FloatTensor')
                targets = targets.type('torch.LongTensor')

                images, targets = images.to(device), targets.to(device)

                gt_pixelwise = torch.ones((2, 64, 64)).type('torch.LongTensor').to(device)
                gt_pixelwise = torch.mul(gt_pixelwise, targets.view(-1, 1, 1))


                scores = net(images)

                loss = criterion(scores, gt_pixelwise)  # scores = (N, 4, 63, 63), gt_pixelwise = (N, 63, 63)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.set_grad_enabled(False):
                    ##for running loss
                    running_loss = 0.1 * running_loss + 0.9 * loss
                    if (i + 1) % 5 == 0:
                        print(
                            'train mode | epoch : [%d/%d] | #fold : [%d/3] | running_loss : [%.5f] iterations : [%d/%d]'
                            % (epoch + 1, whole_epoch, k_fold + 1, running_loss, iter_train, total_iter_train[k_fold]))

                    pred_pixelwise = torch.argmax(scores, dim=1)  # pred_pixelwise = (batch_size, 63, 63), dype = torch.LongTensor

                    ##for accuracy
                    predicts = torch.zeros((2,), dtype=torch.long).to(device)

                    for p in range(2):
                        zero_pixnum = (pred_pixelwise[p] == 0).sum()
                        one_pixnum = (pred_pixelwise[p] == 1).sum()
                        two_pixnum = (pred_pixelwise[p] == 2).sum()
                        thr_pixnum = (pred_pixelwise[p] == 3).sum()
                        predicts[p:p+1] = torch.argmax(
                            torch.IntTensor([zero_pixnum, one_pixnum, two_pixnum, thr_pixnum]), dim=0)

                    true1_false0 = targets == predicts
                    train_acc_one_epoch += true1_false0.sum().item()


                    ##for draw
                    # RGB, pred, GT
                    if i >= len(Dataloader_train) - 5:
                        print('let\'s draw!')
                        rand = np.random.randint(2)
                        rgb = images[rand].cpu().numpy()  # dtype = float, (1024, 1024, 3)
                        pred_pixelwise = pred_pixelwise[rand].cpu().numpy()  # dtype = # int64?, (64, 64)
                        GT = gt_pixelwise[rand].cpu().numpy()  # dtype = long, (64, 64) either 0,1,2,3
                        GT[0][0] = 0
                        GT[0][1] = 1
                        GT[0][2] = 2
                        GT[0][3] = 3
                        fig = plt.figure(figsize=(9, 3))
                        ax1 = fig.add_subplot(131)
                        ax2 = fig.add_subplot(132)
                        ax3 = fig.add_subplot(133)

                        im1 = ax1.imshow(np.transpose(rgb, (1, 2, 0)) / 255)
                        im2 = ax2.imshow(pred_pixelwise, cmap=cmap, norm=norm, interpolation='none')
                        im3 = ax3.imshow(GT, cmap=cmap, norm=norm, interpolation='none')

                        fig.set_constrained_layout_pads(w_pad=2. / 72., h_pad=2. / 72.,
                                                        hspace=0., wspace=0.)
                        # CB = fig.colorbar(ax2, shrink=0.8, extend='both')

                        divider = make_axes_locatable(ax2)
                        cax = divider.append_axes('right', size='3%', pad=0.03)
                        fig.colorbar(im2, cax=cax, orientation='vertical')

                        writer.add_figure('Train|{}fold|{}epoch|last5_iter_figures'.format(k_fold + 1, epoch), fig,
                                          epoch + 1)

            ##for running loss per epoch
            writer.add_scalar('{}_fold_running_loss'.format(k_fold + 1), running_loss, epoch + 1)

            ##for train accuracy per epoch
            train_acc_one_epoch /= len(Dataloader_train.dataset)
            writer.add_scalar('{}fold_train_acc'.format(k_fold + 1), train_acc_one_epoch, epoch + 1)
            train_acc_one_epoch = 0

            confusion_matrixx = torch.zeros(4, 4)

            for i, (images, targets) in enumerate(Dataloader_val):
                iter_val += 1
                net.eval()

                with torch.set_grad_enabled(False):
                    images = images.type('torch.FloatTensor')
                    targets = targets.type('torch.LongTensor')

                    images, targets = images.to(device), targets.to(device)
                    gt_pixelwise = torch.ones((2, 64, 64)).type('torch.LongTensor').to(device)
                    gt_pixelwise = torch.mul(gt_pixelwise, targets.view(-1, 1, 1))

                    scores = net(images)
                    pred_pixelwise = torch.argmax(scores, dim=1)

                    # for accuracy
                    predicts = torch.zeros((2,), dtype=torch.long).to(device)

                    for p in range(2):
                        zero_pixnum = (pred_pixelwise[p] == 0).sum()
                        one_pixnum = (pred_pixelwise[p] == 1).sum()
                        two_pixnum = (pred_pixelwise[p] == 2).sum()
                        thr_pixnum = (pred_pixelwise[p] == 3).sum()
                        predicts[p:p + 1] = torch.argmax(
                            torch.IntTensor([zero_pixnum, one_pixnum, two_pixnum, thr_pixnum]), dim=0)

                    true1_false0 = targets == predicts
                    val_acc_one_epoch += true1_false0.sum().item()

                    print('val mode | epoch : [%d/%d] | #fold : [%d/3] | iterations : [%d/%d]'
                          % (epoch + 1, whole_epoch, k_fold + 1, iter_val, total_iter_val[k_fold]))

                    ##for draw
                    # RGB, pred, GT
                    if i >= len(Dataloader_val) - 5:
                        print('let\'s draw!')
                        rand = np.random.randint(2)
                        rgb = images[rand].cpu().numpy()  # dtype = float, (1024, 1024, 3)
                        pred_pixelwise = pred_pixelwise[rand].cpu().numpy()  # dtype = # int64?, (64, 64)
                        GT = gt_pixelwise[rand].type(
                            'torch.cuda.LongTensor').cpu().numpy()  # dtype = long, (64, 64) either 0,1,2,3
                        GT[0][0] = 0
                        GT[0][1] = 1
                        GT[0][2] = 2
                        GT[0][3] = 3

                        fig = plt.figure(figsize=(9, 3))
                        ax1 = fig.add_subplot(131)
                        ax2 = fig.add_subplot(132)
                        ax3 = fig.add_subplot(133)

                        im1 = ax1.imshow(np.transpose(rgb, (1, 2, 0)) / 255)
                        im2 = ax2.imshow(pred_pixelwise, cmap=cmap, norm=norm, interpolation='none')
                        im3 = ax3.imshow(GT, cmap=cmap, norm=norm, interpolation='none')

                        fig.set_constrained_layout_pads(w_pad=2. / 72., h_pad=2. / 72.,
                                                        hspace=0., wspace=0.)
                        # CB = fig.colorbar(ax2, shrink=0.8, extend='both')

                        divider = make_axes_locatable(ax2)
                        cax = divider.append_axes('right', size='3%', pad=0.03)
                        fig.colorbar(im2, cax=cax, orientation='vertical')

                        writer.add_figure('Val|{}fold|{}epoch|last5_iter_figures'.format(k_fold + 1, epoch), fig,
                                          epoch + 1)

                    # confusion_matrix
                    for t, p in zip(targets.view(-1), predicts.view(-1)):
                        confusion_matrixx[t.long(), p.long()] += 1

            val_acc_one_epoch /= len(Dataloader_val.dataset)
            writer.add_scalar('{}fold_val_acc'.format(k_fold + 1), val_acc_one_epoch, epoch + 1)
            val_acc_whole_epoch.append(val_acc_one_epoch)

            fig = plot_confusion_matrix(confusion_matrixx, classes=['benign', 'cancer1', 'cancer2', 'cancer3'],
                                        title='ConFusionMaTrix')
            writer.add_figure('{}fold confusion_matrix'.format(k_fold + 1), fig, epoch + 1)
            val_acc_one_epoch = 0
            epoch += 1

            # Save the model
            if epoch > 50:
                if not os.path.isdir('./{}/{}fold'.format(Model, k_fold+1)):
                    os.mkdir('./{}/{}fold'.format(Model, k_fold+1))

                PATH = './{}/{}fold/epoch{} valAcc{}.tar'.format(Model, k_fold+1, epoch+1, val_acc_one_epoch)
                torch.save({
                     'epoch': epoch+1,
                     'model_state_dict': net.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict:':scheduler.state_dict()

                }, PATH)

        # PATH = './5_Denseblock_models/{}fold epoch {} valAcc{}.tar'.format(k_fold+1, best_epoch+1, best_val_acc)
        # torch.save({
        #     'epoch': best_epoch+1,
        #     'model_state_dict': best_model_st_dct,
        #     'optimizer_state_dict': best_optimizer_st_dct
        # }, PATH)
        #
        # print('{} Fold\'s best model is saved!')
if __name__ == "__main__":
    main()
