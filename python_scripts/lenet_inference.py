import torch
from torch import nn

class MyLeNet(nn.Module):
    def __init__(self):
        super(MyLeNet, self).__init__()

        self.conv_layer1    = nn.Conv2d(1, 6, 5, 1, 0)
        self.relu_layer1    = nn.ReLU()
        self.avgpool_layer1 = nn.AvgPool2d(2, 2, 0)

        self.conv_layer2    = nn.Conv2d(6, 16, 5, 1, 0)
        self.relu_layer2    = nn.ReLU()
        self.avgpool_layer2 = nn.AvgPool2d(2, 2, 0)


        self.linear_layer1  = nn.Linear(16*5*5, 84)
        self.linear_layer2  = nn.Linear(84, 10)


    def forward(self, input):

        print(f"input shape: {input.shape}")

        output = self.conv_layer1(input)
        output = self.relu_layer1(output)
        print(f"output shape (conv2d layer1): {output.shape}")
        output = self.avgpool_layer1(output)
        print(f"output shape (avgpool layer1): {output.shape}")

        output = self.conv_layer2(output)
        output = self.relu_layer2(output)
        print(f"output shape (conv2d layer2): {output.shape}")
        output = self.avgpool_layer2(output)
        print(f"output shape (avgpool layer2): {output.shape}")

        # transform the volume into a long list of numbers by flatteing
        '''
        flatten = nn.Flatten()
        output  = flatten(output)
        # or the following way
        '''
        output = torch.flatten(output)
        print(f"output shape (after flatteing the pooling volume): {output.shape}")

        output = self.linear_layer1(output)
        print(f"output shape (linear layer1): {output.shape}")
        output = self.linear_layer2(output)
        print(f"output shape (linear layer2): {output.shape}")

        return output


my_lenet = MyLeNet()

input = torch.randn( (1, 1, 32, 32) )
output = my_lenet(input)
print(f"final output shape: {output.shape}")

