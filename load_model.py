from practice import *
from sklearn import metrics
import scikitplot as skplt
if __name__ == '__main__':

    val_whole_y = []
    val_whole_scores = []
    PATH = [
        '/home/hyunwoo/PycharmProjects/pytorch/Apr17_09-31-06_ubuntu/1fold/epoch40 valAcc0.9530864197530864.tar',
        '/home/hyunwoo/PycharmProjects/pytorch/Apr17_09-31-06_ubuntu/2fold/epoch40 valAcc0.9645799011532126.tar',
        '/home/hyunwoo/PycharmProjects/pytorch/Apr17_09-31-06_ubuntu/3fold/epoch40 valAcc0.9522240527182867.tar'
    ]
    for i in [0,1,2]:
        #Dataloader_train = data_loaders[0][0]
        Dataloader_val = data_loaders[i][1]
        #

        net = DenseNet()
        net = nn.DataParallel(net)
        net.to(device)


        PATH_ = PATH[i]
        checkpoint = torch.load(PATH_)

        net.load_state_dict(checkpoint['model_state_dict'])


        confusion_matrixx = torch.zeros(2, 2)
        val_acc_one_epoch = 0



        for i, (images, targets) in enumerate(Dataloader_val):
            net.eval()

            val_whole_y.append(targets.numpy())
            #print('val_while_y is :',val_whole_y)
            images = images.type('torch.FloatTensor')
            targets = torch.tensor(targets)
            images, targets = images.to(device), targets.to(device)



            with torch.set_grad_enabled(False):
                scores = net(images)
                #print('scores : ',scores)
                k = scores.cpu().numpy()
                #print('k : ', k)
                k = k[[0,1,2,3,4,5],[1,1,1,1,1,1]]
                #print('after k :',k)
                val_whole_scores.append(k)
                #print('val_whole+scores :',val_whole_scores)
                predicts = torch.argmax(scores, 1)
                print('predicts :', predicts)
                print('targets :', targets)
                true1_false0 = predicts == targets
                val_acc_one_epoch += true1_false0.sum().item()

                print('val mode | epoch : [%d/40] ' % (checkpoint['epoch'] ))

                # confusion_matrix
                for t, p in zip(targets.view(-1), predicts.view(-1)):
                    confusion_matrixx[t.long(), p.long()] += 1

        val_acc_one_epoch /= len(Dataloader_val.dataset)
        print('val_acc_one_epoch', val_acc_one_epoch)
        print('confusion_matrix :', confusion_matrixx)

    val_whole_y = np.asarray(val_whole_y).flatten()
    val_whole_scores = np.asarray(val_whole_scores).flatten()
    fpr, tpr, thresholds = metrics.roc_curve(val_whole_y, val_whole_scores, pos_label = 1)
    #skplt.metrics.plot_roc_curve(val_whole_y, val_whole_scores)
    print('fpr :', fpr)
    print('tpr :', tpr)
    print('thresholds :', thresholds)
    plt.plot(fpr, tpr)
    plt.show()
    auc_score = metrics.roc_auc_score(val_whole_y, val_whole_scores,)
    print('auc_score ',auc_score)
    #print('fpr :',fpr)
    #print('tpr :',tpr)
    #print('thresholds :',thresholds)
