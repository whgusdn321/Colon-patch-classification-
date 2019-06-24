import torch.nn as nn
from densenet import DenseNet

class NewDensenet(nn.Module):

    def __init__(self):
        super(NewDensenet, self).__init__()
        self.newdense = DenseNet(growth_rate = 12, block_config = (6, 12, 12, 12),\
                                 num_init_features=48, bn_size=4, drop_rate=0)
        num_init_features = 48
        growth_rate = 12
        block_config = [6, 12, 12, 12]

        feature_dim = num_init_features
        feature_dim += growth_rate*block_config[0]
        feature_dim //= 2
        feature_dim += growth_rate * block_config[1]
        feature_dim //= 2
        feature_dim += growth_rate * block_config[2]
        feature_dim //= 2
        feature_dim += growth_rate * block_config[3]

        print('feature_dim ', feature_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.conv_segment = nn.Conv2d(feature_dim, 4, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.newdense(x) # x -> (batch_size, 1024, 128, 128)
        x = self.conv_segment(x)  # x -> (batch_size, 4, 128, 128)
        x = self.softmax(x)  # (batch_size, 4, 128, 128)
        return x