import torch
import torch.nn as nn

EPS = 1e-3

# multi_modal
class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                     padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=True)
    
    def forward(self, x):
        residual = x
        
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, channels=14):
        super(ResNet, self).__init__()
        self.conv_first = nn.Conv2d(channels, 64, kernel_size=3, stride=1,
                padding=1, bias=True)
        self.conv_last = nn.Conv2d(64, channels, kernel_size=1, stride=1,
                padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        self.residual_part = self.make_residual(5)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        
    def make_residual(self, count):
        layers = []
        for _ in range(count):
            layers.append(ResBlock(64, 64))
        return nn.Sequential(*layers)
    
    def forward(self, x2):
        out = self.relu(self.conv_first(x2))
        out = self.residual_part(out)
        out = self.conv_last(out)
        #out = self.sigmoid(out)
        return out
        
        
class ResidualLearningNet(nn.Module):
    def __init__(self, channels=14):
        super(ResidualLearningNet, self).__init__()
        self.residual_layer = self.make_layer(10)
        
        self.input = nn.Conv2d(in_channels=channels, out_channels=64,
                kernel_size=3, stride=1, padding=1, bias=False)
        
        self.output = nn.Conv2d(in_channels=64, out_channels=channels,
                kernel_size=3, stride=1, padding=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                
    def make_layer(self, count):
        layers = []
        for _ in range(count):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64,
                kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x2):
        residual = x2
        out = self.relu(self.input(x2))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out
    
class MRAE(nn.Module):
    def __init__(self):
        super(MRAE, self).__init__()
        
    def forward(self, output, target, mask=None):
        relative_diff = torch.abs(output - target) / (target + 1.0/65535.0)
        if mask is not None:
            relative_diff = mask * relative_diff
        return torch.mean(relative_diff)

class SID(nn.Module):
    def __init__(self):
        super(SID, self).__init__()
        
    def forward(self, output, target, mask=None):
        
        output = torch.clamp(output, 0, 1)
        
        a1 = output * torch.log10((output + EPS) / (target + EPS))
        a2 = target * torch.log10((target + EPS) / (output + EPS))
        
        if mask is not None:
            a1 = a1 * mask
            a2 = a2 * mask
        
        a1_sum = a1.sum(dim=3).sum(dim=2)
        a2_sum = a2.sum(dim=3).sum(dim=2)
        
        errors = torch.abs(a1_sum + a2_sum)
        
        return torch.mean(errors)


# result
class MyNet(nn.Module):
    def __init__(self, channels=1):
        super(MyNet, self).__init__()
        self.residual_layer = self.make_residual_layer(10)
        self.residual_input = nn.Conv2d(in_channels=channels, out_channels=64,
                kernel_size=3, stride=1, padding=1, bias=False)
        self.residual_output = nn.Conv2d(in_channels=64, out_channels=channels,
                kernel_size=3, stride=1, padding=1, bias=False)
        
        self.GCN_layer = self.make_GCN_layer()
        self.GCN_input = nn.Conv1d(in_channels=channels, out_channels=64,
                kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.connected_layer = nn.Linear(in_features=, out_features=)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                
    def make_residual_layer(self, count):
        layers = []
        for _ in range(count):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64,
                kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def make_GCN_layer(self, gcn_layers=[96, 96, 96]):
        layers = []
        layers.append(GraphConv(in_channels, out_channels,
            kernel_size, stride, padding, bias))
        layers.append(nn.ReLU(inplace=True))
        for i in range(1, len(gcn_layers)):
            layers.append(GraphConv(in_channels, out_channels,
                kernel_size, stride, padding, bias))
            layers.append(nn.ReLU(inplace=True))
        # layers = [GraphConv(gcn_layers[0], activation='relu')([hidden_0, hic])]
        # for i in range(1, len(gcn_layers)):
        #     layers.append(GraphConv(gcn_layers[i], activation='relu')([hidden_g[-1], hic]))
        return nn.Sequential(*layers)

    def forward(self, hic_input, epi_input):
        # hic: residual
        residual = hic_input
        out = self.relu(self.residual_input(hic_input))
        out = self.residual_layer(out)
        out = self.residual_output(out)
        residual_out = torch.add(out, residual)
        # epi: GCN
        out = self.relu(self.GCN_input(epi_input))
        GCN_out = self.GCN_layer(out)
        # concatenate
        combined = torch.cat((residual_out, GCN_out), -1)
        # fully connected layer
        out = self.relu(self.connected_layer(combined))
        
        out = torch.add(out, out.t())
        return out