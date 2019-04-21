import sys
sys.path.append("..")
import torch
import torch.nn as nn

def Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None, p=0., bias=True):
    model = [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
    if activation == 'relu':
        model += [nn.ReLU(inplace=True)]

    if p > 0:
        model += [nn.Dropout(p)]
    return nn.Sequential(*model)

def Linear(input_size, output_size, activation=None, p=0., bias=True):
    model = [nn.Linear(input_size, output_size, bias=bias)]
    if activation == 'relu':
        model += [nn.ReLU(inplace=True)]
    elif activation == 'sigmoid':
        model += [nn.Sigmoid()]
    if p > 0.:
        model += [nn.Dropout(p)]
    return nn.Sequential(*model)

def compute_dim(input_size, channels_list, kernel_size_list, strides):
    dim = input_size 
    for i in range(len(strides)):
        dim = int((dim - kernel_size_list[i])/strides[i]) + 1
    dim = dim*channels_list[-1] 
    return dim

class BenchMark(nn.Module):
    """Defination of Variational Autoencoder.

    A simple autoencoder.

    Args:
        input_size: int, the size of input time series.
        channels_list: list, channels of convolutional layers.
        kernel_size_list: list, sizes of kernels.
        strides: list, sizes of strides.
        activation: str, the type of activation.

    Return:
        BenchMark model.
    """ 
    def __init__(self, input_size, output_size, channels_list=[16,32,64], kernel_size_list=[3,3,3], strides=[2,2,2], activation='relu'):
        super(BenchMark, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.out_features = 128
        
        net = []
        for i in range(len(kernel_size_list)):
            if i == 0:
                net += [Conv1d(1, channels_list[i], kernel_size_list[i], stride=strides[i], activation='relu')]
            else:
                net += [Conv1d(channels_list[i-1], channels_list[i], kernel_size_list[i], stride=strides[i], activation='relu')]

        self.conv_layers = nn.Sequential(*net)
        linear_dim = compute_dim(input_size, channels_list, kernel_size_list, strides)
        self.normalize_dim = Linear(linear_dim, self.out_features, 'sigmoid')
        self.fc = Linear(self.out_features, output_size, 'sigmoid')

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.normalize_dim(out)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    benchmark = BenchMark(100, 2)
    print(benchmark)
