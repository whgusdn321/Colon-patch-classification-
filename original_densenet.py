import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Denseblck_element(nn.Module):
    def __init__(self, i, k, input_dim):
        super(Denseblck_element, self).__init__()
        self.batchNo1 = nn.BatchNorm2d(input_dim+k*i)
        self.conv_botlnec = nn.Conv2d(input_dim+k*i, 4*k, 1)
        self.batchNo2 = nn.BatchNorm2d(4*k)
        self.conv_last = nn.Conv2d(4*k,k,3, padding = 1)

    def forward(self, x):
        x = self.batchNo1(x)
        x = F.relu(x)
        x = self.conv_botlnec(x)
        x = self.batchNo2(x)
        x = F.relu(x)
        x = self.conv_last(x)
        return x


class Denseblck_Module(nn.Module):

    def __init__(self, k, L, input_dim):
        super(Denseblck_Module, self).__init__()
        self.Denses = nn.ModuleList([Denseblck_element(i, k, input_dim) for i in range(L)])
        self.L = L
    def forward(self, x):
        x_concated = x
        for i in range(self.L):
            x = self.Denses[i](x_concated)
            x_concated = torch.cat((x_concated, x), 1)
        return x_concated


class Transition_Module(nn.Module):
    def __init__(self, input_dim):
        super(Transition_Module, self).__init__()
        self.conv_transition = nn.Conv2d(input_dim, input_dim//2, 1)
        nn.init.xavier_normal_(self.conv_transition.weight)
        self.avg_pool = nn.AvgPool2d(2, stride = 2)

    def forward(self, x):
        x = self.conv_transition(x)
        x = self.avg_pool(x)
        return x

class DenseNet1(nn.Module):
    '''
       Dense net- BC version

       structure :
       conv(7*7*2k/stride:2, pad : 3)---maxpooling(3*3, stride:2)---{Denseblock(L, k)---Transition(theta)} X2---Denseblock(L, K)---averagepool(7,7, stride :1)---affine---softmax

       hyper parameters : L, k, theta.
       L : num of layers in Dense block
       k : growth tate ->  num of filters of layers in Dense block. This means, each output of Dense block will have 'k' feature maps
       theta : compression factor. used in bottleneck, and transition layers. This means, to reduce the number of feaure-maps of
               Dense block input/output, conv layer in transition-Layer(conv,maxpool) will reduce it by convolving with ( 1 X 1 X k * gamma ), 0 < gamma <=1

       #L = 100,
       #K = 12
       #theta = 0.5
       '''

    '''
    def denseblock:
        #for example. when L = 3 , k = 10
        #convolution one time
        # 
    '''
    def __init__(self):
        super(DenseNet1, self).__init__()
        k = 12
        L = 15
        #--------Conv Layer and pooling layer before Denseblock---------
        self.conv1 = nn.Conv2d(3, 2*k, 7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(3, stride=2)
        '''---------------------------------------------------------------'''

        self.dense1 = Denseblck_Module(k, L, input_dim=2*k) # k -> Dense element output activatoion map length
                                                            # L -> Num of Dense elements
                                                            # input_dim = 2*k( output size from conv1,pool )
        output_dim = 2*k+ k*L
        #-----------------------transition layer1------------------------
        self.trasition1 = Transition_Module(input_dim=output_dim)
        output_dim //= 2
        #---------------------------------------------------------------

        self.dense2 = Denseblck_Module(k, L, input_dim=output_dim)
        output_dim = output_dim + k*L

        #----------------------transition layer2------------------------
        self.trasition2 = Transition_Module(input_dim=output_dim)
        output_dim //= 2
        #---------------------------------------------------------------

        self.dense3 = Denseblck_Module(k, L, input_dim=output_dim)
        output_dim = output_dim + k*L

        # #add one more transition and dense_blck
        # self.trasition3 = Transition_Module(input_dim=output_dim)
        # output_dim //= 2
        #
        # self.dense4 = Denseblck_Module(k, L, input_dim=output_dim)
        # output_dim = output_dim + k*L
        #
        # ## add one more transition and dsn_blck
        # self.trasition4 = Transition_Module(input_dim=output_dim)
        # output_dim //= 2
        # #
        # self.dense5 = Denseblck_Module(k, L, input_dim=output_dim)
        # output_dim = output_dim + k * L

        #-----------------------global_avg_pool--------------------------
        self.bn = nn.BatchNorm2d(output_dim)#->

        self.conv_segment = nn.Conv2d(321, 4, 1, stride=1, padding=0) #->[3, 4, 63, 63]



        # self.global_avgpool = nn.AvgPool2d(63, stride = 1)
        # self.fully_connected = nn.Linear(output_dim, 4)
        #nn.init.xavier_normal_(self.fully_connected.weight)

        #----------------------log_softmax-------------------------------
        self.softmax = nn.LogSoftmax(dim=1)

        for m in self.modules():
            torch.cuda.manual_seed(100)
            if isinstance(m, nn.Conv2d):
                 n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                 m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                 m.weight.data.fill_(1)
                 m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                 m.bias.data.zero_()

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool(x)
        x = self.dense1(x)
        x = self.trasition1(x)
        x = self.dense2(x)
        x = self.trasition2(x)
        x = self.dense3(x)

        # x = self.trasition3(x)
        # x = self.dense4(x)
        # # # #
        # x = self.trasition4(x)
        # x = self.dense5(x)

        x = F.relu(self.bn(x))
        x = self.conv_segment(x)
        # print('x.shape is ', x.size())
        # x = self.global_avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fully_connected(x)
        x = self.softmax(x) #->should be shape [3,4,63,63]

        return x
