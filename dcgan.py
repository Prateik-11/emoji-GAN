import torch


class Generator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=False)
        self.tanh = torch.nn.Tanh()
        self.conv1 = torch.nn.ConvTranspose2d(in_channels=100, 
                                    out_channels=512,
                                    kernel_size=4,
                                    stride=1,
                                    padding=0,
                                    bias=False
                                    )
        self.bn1 = torch.nn.BatchNorm2d(num_features=512)
        self.conv2 = torch.nn.ConvTranspose2d(in_channels=512, 
                                    out_channels=256,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    bias=False
                                    )
        self.bn2 = torch.nn.BatchNorm2d(num_features=256)
        self.conv3 = torch.nn.ConvTranspose2d(in_channels=256, 
                                    out_channels=128,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    bias=False
                                    )
        self.bn3 = torch.nn.BatchNorm2d(num_features=128)
        self.conv4 = torch.nn.ConvTranspose2d(in_channels=128, 
                                    out_channels=64,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    bias=False
                                    )
        self.bn4 = torch.nn.BatchNorm2d(num_features=64)
        self.conv5 = torch.nn.ConvTranspose2d(in_channels=64,
                                    out_channels=3,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    bias=False
                                    )
        self.weights_init()

    def forward(self, z):
        z = self.relu(self.bn1(self.conv1(z)))
        z = self.relu(self.bn2(self.conv2(z)))
        z = self.relu(self.bn3(self.conv3(z)))
        z = self.relu(self.bn4(self.conv4(z)))
        z = self.tanh(self.conv5(z))

        return z
    
    def weights_init(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            torch.nn.init.normal_(layer.weight.data, 0.0, 0.02)
        for layer in [self.bn1, self.bn2, self.bn3]:
            torch.nn.init.normal_(layer.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(layer.bias.data, 0)

class Discriminator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, 
                                            inplace=False
                                            )
        self.sigmoid = torch.nn.Sigmoid()
        
        self.conv1 = torch.nn.Conv2d(in_channels=3,
                                    out_channels=64,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    bias=False
                                    )
        self.conv2 = torch.nn.Conv2d(in_channels=64,
                                    out_channels=128,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    bias=False
                                    )
        self.bn1 = torch.nn.BatchNorm2d(num_features=128)
        self.conv3 = torch.nn.Conv2d(in_channels=128,
                                    out_channels=256,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    bias=False
                                    )
        self.bn2 = torch.nn.BatchNorm2d(num_features=256)
        self.conv4 = torch.nn.Conv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    bias=False
                                    )
        self.bn3 = torch.nn.BatchNorm2d(num_features=512)
        self.conv5 = torch.nn.Conv2d(in_channels=512,
                                    out_channels=1,
                                    kernel_size=4,
                                    stride=1,
                                    padding=0,
                                    bias=False
                                    )
        self.weights_init()

    def forward(self, image):
        x = self.leaky_relu(self.conv1(image))
        x = self.leaky_relu(self.bn1(self.conv2(x)))
        x = self.leaky_relu(self.bn2(self.conv3(x)))
        x = self.leaky_relu(self.bn3(self.conv4(x)))
        x = self.conv5(x)
        x = self.sigmoid(self.conv5(x))
        
        return x

    def weights_init(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            torch.nn.init.normal_(layer.weight.data, 0.0, 0.02)
        for layer in [self.bn1, self.bn2, self.bn3]:
            torch.nn.init.normal_(layer.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(layer.bias.data, 0)

# to do:
# increase the model, decrease the image size
# move everything on colab
# copy stylegan2 architecture
# 