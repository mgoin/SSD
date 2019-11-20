import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url

# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-b0353104.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-b0353104.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-b0353104.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-b0353104.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b0353104.pth',
# }


class ResNet_Base(nn.Module):
    def __init__(self, backbone='resnet50', backbone_path=None):
        super(ResNet_Base, self).__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=not backbone_path)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=not backbone_path)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet152':
            backbone = resnet152(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:
            raise ValueError("Invalid resnet backbone:", backbone)

        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class ResNet(nn.Module):
    def __init__(self, backbone="resnet50"):
        super(ResNet, self).__init__()

        self.backbone = backbone
        self.feature_extractor = ResNet_Base(backbone)

        self.label_num = 81  # number of COCO classes
        self._build_additional_features(self.feature_extractor.out_channels)

        self._init_weights()

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3,
                              padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size,
                              kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, x):
        print("ResNet input:\n", x)
        x = self.feature_extractor(x)

        features = [x]
        for l in self.additional_blocks:
            x = l(x)
            features.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        print("ResNet features:\n", tuple(features))
        return tuple(features)


@registry.BACKBONES.register('resnet18')
def resnet18(cfg, pretrained=True):
    model = ResNet('resnet18')
    # if pretrained:
    #     unexpected = model.load_state_dict(load_state_dict_from_url(
    #         model_urls['resnet18']), strict=False)
    #     print("Unexpected weights:", unexpected)
    return model


@registry.BACKBONES.register('resnet34')
def resnet34(cfg, pretrained=True):
    model = ResNet('resnet34')
    # if pretrained:
    #     unexpected = model.load_state_dict(load_state_dict_from_url(
    #         model_urls['resnet34']), strict=False)
    #     print("Unexpected weights:", unexpected)
    return model


@registry.BACKBONES.register('resnet50')
def resnet50(cfg, pretrained=True):
    model = ResNet('resnet50')
    # if pretrained:
    #     unexpected = model.load_state_dict(load_state_dict_from_url(
    #         model_urls['resnet50']), strict=False)
    #     print("Unexpected weights:", unexpected)
    return model


@registry.BACKBONES.register('resnet101')
def resnet101(cfg, pretrained=True):
    model = ResNet('resnet101')
    # if pretrained:
    #     unexpected = model.load_state_dict(load_state_dict_from_url(
    #         model_urls['resnet101']), strict=False)
    #     print("Unexpected weights:", unexpected)
    return model


@registry.BACKBONES.register('resnet152')
def resnet152(cfg, pretrained=True):
    model = ResNet('resnet152')
    # if pretrained:
    #     unexpected = model.load_state_dict(load_state_dict_from_url(
    #         model_urls['resnet152']), strict=False)
    #     print("Unexpected weights:", unexpected)
    return model
