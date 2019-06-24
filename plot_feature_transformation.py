"""
===============================================
Feature transformations with ensembles of trees
===============================================


Transform your features into a higher dimensional, sparse space. Then
train a linear model on these features.

First fit an ensemble of trees (totally random trees, a random
forest, or gradient boosted trees) on the training set. Then each leaf
of each tree in the ensemble is assigned a fixed arbitrary feature
index in a new feature space. These leaf indices are then encoded in a
one-hot fashion.

Each sample goes through the decisions of each tree of the ensemble
and ends up in one leaf per tree. The sample is encoded by setting
feature values for these leaves to 1 and the other feature values to 0.

The resulting transformer has then learned a supervised, sparse,
high-dimensional categorical embedding of the data.

"""

# Author: Tim Head <betatim@gmail.com>
#
# License: BSD 3 clause

import numpy as np
np.random.seed(10)

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline





from practice import *
from sklearn import metrics
import scikitplot as skplt
if __name__ == '__main__':


    # val_whole_y = []
    # val_whole_scores = []
    # PATH = [
    #      '/home/hyunwoo/PycharmProjects/pytorch/Apr13_09-18-14_ubuntu/1fold/epoch40 valAcc0.9786008230452675.tar',
    #     '/home/hyunwoo/PycharmProjects/pytorch/Apr13_09-18-14_ubuntu/2fold/epoch40 valAcc0.9876441515650741.tar',
    #     '/home/hyunwoo/PycharmProjects/pytorch/Apr13_09-18-14_ubuntu/3fold/epoch40 valAcc0.9744645799011532.tar'
    #
    # ]
    # for i in [0,1,2]:
    #     #Dataloader_train = data_loaders[0][0]
    #     Dataloader_val = data_loaders[i][1]
    #     #
    #
    #     net = DenseNet()
    #     net = nn.DataParallel(net)
    #     net.to(device)
    #
    #
    #     PATH_ = PATH[i]
    #     checkpoint = torch.load(PATH_)
    #
    #     net.load_state_dict(checkpoint['model_state_dict'])
    #
    #
    #     confusion_matrixx = torch.zeros(2, 2)
    #     val_acc_one_epoch = 0
    #
    #
    #
    #     for i, (images, targets) in enumerate(Dataloader_val):
    #         net.eval()
    #
    #         val_whole_y.append(targets.numpy())
    #         #print('val_while_y is :',val_whole_y)
    #         images = images.type('torch.FloatTensor')
    #         targets = torch.tensor(targets)
    #         images, targets = images.to(device), targets.to(device)
    #
    #
    #
    #         with torch.set_grad_enabled(False):
    #             scores = net(images)
    #             #print('scores : ',scores)
    #             k = scores.cpu().numpy()
    #             #print('k : ', k)
    #             k = k[[0,1,2,3,4,5],[1,1,1,1,1,1]]
    #             #print('after k :',k)
    #             val_whole_scores.append(k)
    #             #print('val_whole+scores :',val_whole_scores)
    #             predicts = torch.argmax(scores, 1)
    #             print('predicts :', predicts)
    #             print('targets :', targets)
    #             true1_false0 = predicts == targets
    #             val_acc_one_epoch += true1_false0.sum().item()
    #
    #             print('val mode | epoch : [%d/40] ' % (checkpoint['epoch'] ))
    #
    #             # confusion_matrix
    #             for t, p in zip(targets.view(-1), predicts.view(-1)):
    #                 confusion_matrixx[t.long(), p.long()] += 1
    #
    #     val_acc_one_epoch /= len(Dataloader_val.dataset)
    #     print('val_acc_one_epoch', val_acc_one_epoch)
    #     print('confusion_matrix :', confusion_matrixx)
    #
    # val_whole_y = np.asarray(val_whole_y).flatten()
    # val_whole_scores = np.asarray(val_whole_scores).flatten()
    # fpr, tpr, thresholds = metrics.roc_curve(val_whole_y, val_whole_scores, pos_label = 1)
    #
    # if not os.path.isdir('./{}'.format('1024_3_Densblkmodel_fpr_tpr')):
    #     os.mkdir('./{}'.format('1024_3_Densblkmodel_fpr_tpr'))
    #
    # PATH = './{}/ {}.tar'.format('1024_3_Densblkmodel_fpr_tpr', '1024_3_Densblkmodel')
    # torch.save({
    #     'fpr': fpr,
    #     'tpr': tpr,
    #     'thresholds': thresholds,
    #     'label': '1024_3_Densblkmodel'
    #
    # }, PATH)




    TPR_FPR_PATH_512 = [
       #Total 6 path
        '/home/hyunwoo/PycharmProjects/pytorch/512_3_Densblkmodel_fpr_tpr/512_3_Densblkmodel.tar', #512_3Dbl
        '/home/hyunwoo/PycharmProjects/pytorch/512_4_Densblkmodel_fpr_tpr/512_4_Densblkmodel.tar', #512_4Dbl
        '/home/hyunwoo/PycharmProjects/pytorch/512_5_Densblkmodel_fpr_tpr/512_5_Densblkmodel.tar' #512_5Dbl

    ]
    TPR_FPR_PATH_1024 =[
        '/home/hyunwoo/PycharmProjects/pytorch/1024_3_Densblkmodel_fpr_tpr/ 1024_3_Densblkmodel.tar',  # 1024_3Dbl
        '/home/hyunwoo/PycharmProjects/pytorch/1024_4_Densblkmodel_fpr_tpr/ 1024_4_Densblkmodel.tar',  # 1024_4Dbl
        '/home/hyunwoo/PycharmProjects/pytorch/1024_5_Densblkmodel_fpr_tpr/ 1024_5_Densblkmodel.tar'  # 1024_5Dbl
    ]

    fpr_512 = []
    tpr_512 = []
    fpr_1024 = []
    tpr_1024 = []
    #thresholds = []
    label512 = ['512x512 3DB-TL', '512x512 4DB-TL', '512x512 5DB-TL']
    label1024 = ['1024x1024 3DB-TL', '1024x1024 4DB-TL', '1024x1024 5DB-TL']

    for path in TPR_FPR_PATH_512:
        checkpoint = torch.load(path)
        fpr_512.append(checkpoint['fpr'])
        tpr_512.append(checkpoint['tpr'])
        #thresholds.append(checkpoint['thresholds'])

    for path in TPR_FPR_PATH_1024:
        checkpoint = torch.load(path)
        fpr_1024.append(checkpoint['fpr'])
        tpr_1024.append(checkpoint['tpr'])
    #
    # color = ['red', 'blue', 'green']
    # plt.figure(4)

    # for fpr, tpr, label_1024, col in zip(fpr_1024, tpr_1024, label1024, color):
    #     plt.plot(fpr, tpr, label=label_1024, color = col)
    # for fpr,tpr, label_512, col in zip(fpr_512, tpr_512, label512, color):
    #     plt.plot(fpr, tpr, dashes=[2, 2], label=label_512, color = col, )
    #
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # #plt.legend(loc='best')
    # plt.show()
    # plt.pause(50)

    color = ['red', 'blue', 'green']
    plt.figure(6)
    plt.xlim(0, 0.1)
    plt.ylim(0.9, 1)

    for fpr, tpr, label_1024, col in zip(fpr_1024, tpr_1024, label1024, color):
        plt.plot(fpr, tpr, label = label_1024, color = col)

    for fpr, tpr,  label_512, col in zip(fpr_512, tpr_512, label512, color):
        plt.plot(fpr, tpr, - 0.2, dashes=[3, 3], label=label_512, color = col)

    plt.xticks(np.arange(0, 0.1+0.000001 , 0.05), fontsize = 16)
    plt.yticks(fontsize = 16)
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    #plt.title('ROC curve (zoomed in at top left)')
    #plt.legend(loc='best')
    plt.show()
    plt.pause(50)



















#
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
# plt.plot(fpr_rf, tpr_rf, label='RF')
# plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
# plt.plot(fpr_grd, tpr_grd, label='GBT')
# plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure(2)
# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
# plt.plot(fpr_rf, tpr_rf, label='RF')
# plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
# plt.plot(fpr_grd, tpr_grd, label='GBT')
# plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve (zoomed in at top left)')
# plt.legend(loc='best')
# plt.show()
