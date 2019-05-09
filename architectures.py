from core import *
from utils import load_model
# from __utils__ import *
from data_manipulation import *

torch.manual_seed(42)
torch.cuda.manual_seed(42)

###################################
######### 1st set of experiments####
###################################


def get_top_layers(pretrained):
    '''Return a list of the top groups of paramteres.
    Handles cases when:
    > Model structure is in sequential groups (MURA)
    > Model structure is the default from densenet121 architecture
    '''
    if pretrained == 'MURA':
        top = DenseNet121(1, False)
        load_model(top, '../latest_models/mura_2.pth')
        out = list([group.children() for group in top.groups[:2]])
    elif pretrained in (True, False):
        top_model = models.densenet121(pretrained=pretrained)
        top_layers = list(top_model.children())[0]
        out = [top_layers[:7], top_layers[7:]]

    else:
        raise ValueError("Invalid pretrained value")

    return out


class DenseNet121(nn.Module):
    '''
    DenseNet121 with quick iterations on:
     > arbitrary finite out_size.
     > pre-trained model between ImageNet and the medical image data-set MURA (all but last layer).
     > freeze layers
    '''

    def __init__(self, out_size: int = 14, pretrained: bool = False, freeze: str = False):
        '''

        :param out_size: (int) output size
        :param pretrained: (bool/str) Kind of pre-train: Supports  'MURA', True and False.
        :param freeze: (bool) freeze all layers but last one.
        '''
        super().__init__()

        top_layers_groups = get_top_layers(pretrained)

        self.groups = nn.ModuleList([nn.Sequential(*group) for group in top_layers_groups])
        self.groups.append(nn.Linear(1024, out_size))

        if freeze: self.freeze([0, 1])

    def forward(self, x):

        for group in self.groups[:-1]:
            x = group(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)
        x = self.groups[-1](x)
        return x

    def freeze(self, group_idxs: (list, int, tuple)):
        if not isinstance(group_idxs, (list, tuple)): group_idxs = [group_idxs]
        for group_idx in group_idxs:
            group = self.groups[group_idx]
            parameters = filter(lambda x: x.requires_grad, group.parameters())
            for p in parameters:
                p.requires_grad = False

    def unfreeze(self, group_idx: int):
        if group_idx not in [0, 1, 2]: raise ValueError('group_idx must be between 0 and 2')
        group = self.groups[group_idx]
        parameters = filter(lambda x: hasattr(x, 'requires_grad'), group.parameters())
        for p in parameters: p.requires_grad = True

#######################################
######## synthetic Data (CNN) #########
#######################################

class adapt_to_problem(nn.Module):
    """
    Take an architecture from the torchvision modules and automatically
    adapts it to your problem.

    Currently supports DensNets and ResNets.
    """

    def __init__(self, model, out_dim, pretrained=True, freeze=True):
        '''
        :param model:       torchvision model class. e.g: torchvison.models.resnet50
        :param out_dim:     Number of categorical/continues variables in the target.
                            e.g : In the ChestXRay dataset it would be 14.
        :param pretrained:  Bool
        :param freeze:      Bool. Freezes all layers but the last one in the model.
        '''
        super().__init__()
        self.in_features = list(model().children())[-1].in_features
        self.layers = list(model(pretrained=pretrained).children())[:-1]

        self.top_model = nn.Sequential(*self.layers)
        self.linear = nn.Linear(self.in_features, out_features=out_dim, bias=True)

        if 'densenet' in model.__name__:
            self.forward = self.forward_densnet
        elif 'resnet' in model.__name__:
            self.forward = self.forward_resnet
        else:
            raise NotImplementedError

        if freeze: self.freeze_top_model()

    def forward_resnet(self, x):

        out = self.top_model(x).squeeze()
        out = self.linear(out)

        return out

    def forward_densnet(self, x):

        features = self.top_model(x)
        features = F.relu(features)
        out = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        out = self.linear(out)

        return out

    def freeze_top_model(self):
        for p in self.top_model.parameters():
            p.requires_grad = False


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """
    1x1 convolution
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class meta_generator(nn.Module):
    '''
    Creates a NN composed by ResNet blocks with # bloks = len(planes_per_block) and the number of
    channels per in each block specified in planes_per_block for an arbitrary output size out_size.


    Includes a functionality (as attribute) to quickly get the # of trainable parameters in the net.
    '''

    def __init__(self, inplanes: int, planes_per_block: list, out_size: int = 1):
        '''

        :param inplanes: (int) # of planes in the input (RGB -3-/Grey -1- image)
        :param planes_per_block: (list) sequential amount of channels in each block.
                                The length of the list is the # of blocks.
        :param out_size: (int) output size.
        '''
        super().__init__()

        self.conv1 = conv1x1(inplanes, planes_per_block[0])
        self.bn1 = nn.BatchNorm2d(planes_per_block[0])

        self.block_seq = nn.Sequential()

        for i in range(len(planes_per_block) - 1):
            # block
            self.block_seq.add_module(f'block{str(i)}', BasicBlock(planes_per_block[i], planes_per_block[i]))

            # inter-block transition
            self.block_seq.add_module(f'conv_transition{str(i)}', conv1x1(planes_per_block[i], planes_per_block[i + 1]))
            self.block_seq.add_module(f'bn_transition{str(i)}', nn.BatchNorm2d(planes_per_block[i + 1]))
            self.block_seq.add_module(f'relu_transition{str(i)}', nn.ReLU(inplace=True))

        self.block_seq.add_module(f'block{str(len(planes_per_block)-1)}',
                                  BasicBlock(planes_per_block[-1], planes_per_block[-1]))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lin = nn.Linear(planes_per_block[-1], out_size)

        self.n_parameters = self.get_n_parameters()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.block_seq(out)

        out = self.avgpool(out)
        out = out.squeeze()
        out = self.lin(out)

        return out

    def get_n_parameters(self):
        return sum([np.prod(list(p.size())) for p in filter(lambda x: x.requires_grad, self.parameters())])


class conv_batch_act(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_sz, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_sz, stride)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class cnn(nn.Module):
    '''
    Creates a CNN with six layers

        7x7 with stride 2 and 1 channel out
        batchnorm
        relu
        3x3 with stride 2 and 2 channel out
        batchnorm
        relu

    Followed by the number of layers with:

        3x3 with stride 2

    and the number of channels specified by the channels list

    The last layer is a linear after doing an Adaptative avg pooling.
    '''

    def __init__(self, channels: list, out_size: int = 1, inplanes: int = 3):
        '''
        :param channels: list of channels for intermediate convolutions
        :param out_size: size of the output
        '''
        super().__init__()
        if len(channels) > 0:
            if channels[-1] == 1: warnings.warn("last layer is a 1-in 1-out linear", UserWarning)

        layers = []

        layers.append(conv_batch_act(inplanes, 2, 7, 2))

        layers.append(conv_batch_act(2, 4, 3, 2))

        prev_ch = 4

        # if len(channels)>0:

        for element in channels:
            layers.append(conv_batch_act(prev_ch, element, 3, 1))
            prev_ch = element

        self.top_model = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.lin = nn.Linear(prev_ch, out_size)

    def forward(self, x):

        out = self.top_model(x)

        out = self.pool(out).squeeze()

        out = self.lin(out)

        return out

    def get_n_parameters(self):
        return sum([np.prod(p.size()) for p in filter(lambda x: x.requires_grad, self.parameters())])


#############################################
#### Other work during experimentation ######
#############################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        #         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #         self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        #         x = self.layer2(x)
        #         x = self.layer3(x)
        #         x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = x.squeeze()

        return x


def Convolutions(mini_blocks: int = 2, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [mini_blocks], **kwargs).cuda()

    return model


class model_ResNet18(nn.Module):

    def __init__(self, out_size=1, pretrained=False):
        super(model_ResNet18, self).__init__()

        self.resnet18 = models.resnet18(pretrained=pretrained)
        layers = list(self.resnet18.children())[:-2]
        self.top_model = nn.Sequential(*layers).cuda()

        # self.avgpool = nn.AvgPool2d(kernel_size=(8, 8), stride=1, padding=0)

        self.linear = nn.Linear(512, out_size)

    def forward(self, x):
        x = F.relu(self.top_model(x))
        # x = self.avgpool(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

    def half(self):
        for module in self.children():
            if type(module) is not nn.BatchNorm2d:
                module.half()
        return self


class MURAResnet50_1layer(nn.Module):
    def __init__(self, pretrained=False):
        super(MURAResnet50_1layer, self).__init__()
        RN50 = models.resnet50(pretrained=pretrained)
        self.top_model = nn.Sequential(*list(RN50.children())[:-1]).cuda()

        self.fc1 = nn.Linear(in_features=2048, out_features=1)

    def forward(self, x):
        x = self.top_model(x)

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)

        return x

    def half(self):
        for module in self.children():
            if type(module) is not nn.BatchNorm2d:
                module.half()
        return self

    def first_block_freeze(self):
        for param in self.top_model.parameters():
            param.requires_grad = False

    def first_block_unfreeze(self):
        for param in self.top_model.parameters():
            param.requires_grad = True


class MURAResnet50_3layers(nn.Module):
    def __init__(self, p=.2, pretrained=False):
        super(MURAResnet50_3layers, self).__init__()
        RN50 = models.resnet50(pretrained=pretrained)
        self.top_model = nn.Sequential(*list(RN50.children())[:-1]).cuda()

        self.fc1 = nn.Linear(in_features=2048, out_features=1000)

        self.act1 = nn.ReLU()

        self.bn2 = nn.BatchNorm1d(1000)
        self.d2 = nn.Dropout(p)
        self.fc2 = nn.Linear(1000, 250)
        self.act2 = nn.ReLU()

        self.bn3 = nn.BatchNorm1d(250)
        self.d3 = nn.Dropout(p)
        self.fc3 = nn.Linear(250, 1)

        # act1 = nn.Sigmoid

    def forward(self, x):
        x = self.top_model(x)

        x = x.view(x.shape[0], -1)

        x = self.act1(self.fc1(x))

        x = self.act2(self.fc2(self.d2(self.bn2(x))))

        x = self.fc3(self.d3(self.bn3(x)))

        return x

    def half(self):
        for module in self.children():
            if type(module) is not nn.BatchNorm2d:
                module.half()
        return self

    def first_block_freeze(self):
        for param in self.top_model.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.bn2.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = False
        for param in self.bn3.parameters():
            param.requires_grad = False
        for param in self.fc3.parameters():
            param.requires_grad = False

    def first_block_unfreeze(self):
        for param in self.top_model.parameters():
            param.requires_grad = True
        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.bn2.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = True
        for param in self.bn3.parameters():
            param.requires_grad = True
        for param in self.fc3.parameters():
            param.requires_grad = True
