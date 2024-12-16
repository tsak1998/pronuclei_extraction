import torch
import torch.nn as nn
from torchvision import models


class conv_block(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class up_conv(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):

    def __init__(self, img_ch=3, output_ch=1):
        super(UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,
                                  output_ch,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class UNetWithPretrainedEncoder(nn.Module):

    def __init__(self, img_ch=3, output_ch=1, encoder_name="resnet34"):
        super(UNetWithPretrainedEncoder, self).__init__()

        # Load pretrained encoder
        if encoder_name == "resnet34":
            self.encoder = models.resnet34(pretrained=True)
            encoder_channels = [64, 64, 128, 256,
                                512]  # Output channels for ResNet34
        elif encoder_name == "efficientnet_b0":
            self.encoder = models.efficientnet_b0(pretrained=True)
            encoder_channels = [32, 24, 40, 112,
                                320]  # Output channels for EfficientNetB0
        else:
            raise ValueError("Unsupported encoder type")

        # Freeze encoder weights if needed
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder layers
        self.Up5 = up_conv(ch_in=512, ch_out=256)  # Upsampling from x5
        self.Up_conv5 = conv_block(ch_in=512 + 512, ch_out=256)  # Concatenate x4 and d5

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256 + 256, ch_out=128)  # Concatenate x3 and d4

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128 + 128, ch_out=64)  # Concatenate x2 and d3

        self.Up2 = up_conv(ch_in=64, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=64 + 64, ch_out=64)    # Concatenate x1 and d2

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # Encoding path using the pretrained encoder
        if isinstance(self.encoder, models.ResNet):
            x1 = self.encoder.relu(self.encoder.bn1(
                self.encoder.conv1(x)))  # Stage 1
            x2 = self.encoder.layer1(self.Maxpool(x1))  # Stage 2
            x3 = self.encoder.layer2(x2)  # Stage 3
            x4 = self.encoder.layer3(x3)  # Stage 4
            x5 = self.encoder.layer4(x4)  # Stage 5
        elif isinstance(self.encoder, models.EfficientNet):
            features = self.encoder.features(x)
            x1, x2, x3, x4, x5 = features[0], features[1], features[
                2], features[3], features[4]

        # Decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttUNet(nn.Module):

    def __init__(self, output_ch=1, pretrained=True):
        super(AttUNet, self).__init__()

        # Load pretrained ResNet model
        resnet = models.resnet34(pretrained=pretrained)

        # Extract layers from ResNet
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1,
                                    resnet.relu)  # 64 channels
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        # Decoder
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,
                                  output_ch,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x):
        # Encoder with ResNet layers
        x1 = self.layer0(x)  # 64 channels
        x2 = self.layer1(x1)  # 64 channels
        x3 = self.layer2(x2)  # 128 channels
        x4 = self.layer3(x3)  # 256 channels
        x5 = self.layer4(x4)  # 512 channels

        # Bottleneck
        x5 = self.Conv5(x5)

        # Decoder + Attention + Skip Connections
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
