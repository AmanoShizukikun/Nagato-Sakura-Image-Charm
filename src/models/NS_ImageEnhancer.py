import torch
import torch.nn as nn


class RRDB(nn.Module):
    def __init__(self, in_channels, growth_channels=32):
        super(RRDB, self).__init__()
        self.dense1 = nn.Sequential(
            nn.Conv2d(in_channels, growth_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dense2 = nn.Sequential(
            nn.Conv2d(in_channels + growth_channels, growth_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dense3 = nn.Sequential(
            nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_conv = nn.Conv2d(in_channels + 3 * growth_channels, in_channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out1 = self.dense1(x)
        out = torch.cat([x, out1], dim=1)
        out2 = self.dense2(out)
        out = torch.cat([out, out2], dim=1)
        out3 = self.dense3(out)
        out = torch.cat([out, out3], dim=1)
        out = self.final_conv(out)
        return self.lrelu(residual + out * 0.2)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out

class ImageQualityEnhancer(nn.Module):
    def __init__(self, num_rrdb_blocks=16, features=64):
        super(ImageQualityEnhancer, self).__init__()
        self.conv_first = nn.Conv2d(3, features, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.rrdb_blocks = nn.ModuleList([RRDB(features * 4) for _ in range(num_rrdb_blocks)])
        self.attention = AttentionBlock(features * 4)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 4, features * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 4, features * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(features * 4, features * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        initial_features = self.conv_first(x)
        encoder_out = self.encoder(initial_features)
        rrdb_out = encoder_out
        for rrdb in self.rrdb_blocks:
            rrdb_out = rrdb(rrdb_out)
        attention_out = self.attention(rrdb_out)
        upsampled = self.upsample(attention_out)
        out = self.final_conv(upsampled)
        return out