import torch.nn as nn
import torch
# import torch.optim as optim


class LearnableSigmoid(torch.nn.Module):
    def __init__(self, ):
        super(LearnableSigmoid, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(1.0)

    def forward(self, input):
        return 1 / (1 + torch.exp(-self.weight * input))


# Sigmoid = torch.nn.Sigmoid()
# LearnSigmoid = LearnableSigmoid()
# input = torch.tensor([[0.5289, 0.1338, 0.3513],
#                       [0.4379, 0.1828, 0.4629],
#                       [0.4302, 0.1358, 0.4180]])
#
# print(Sigmoid(input))
# print(LearnSigmoid(input))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.LSigmoid = LearnableSigmoid()

    def forward(self, x):
        x = self.LSigmoid(x)
        return x

#
# net = Net()
# print(list(net.parameters()))
# optimizer = optim.SGD(net.parameters(), lr=0.01)
# learning_rate = 0.001
# input_data = torch.randn(10, 2)
# target = torch.FloatTensor(10, 2).random_(8)
# criterion = torch.nn.MSELoss(reduce=True, size_average=True)
#
# for i in range(2):
#     optimizer.zero_grad()
#     output = net(input_data)
#     loss = criterion(output, target)
#     loss.backward()
#     optimizer.step()
#     print(list(net.parameters()))