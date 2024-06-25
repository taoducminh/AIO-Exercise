import torch
import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x)

class SoftmaxStable(nn.Module):
    def __init__(self):
        super(SoftmaxStable, self).__init__()

    def forward(self, x):
        max_x = torch.max(x)
        exp_x = torch.exp(x - max_x)  # subtract max_x for numerical stability
        return exp_x / torch.sum(exp_x)

# Examples
data = torch.Tensor([1, 2, 3])

# Using Softmax
softmax = Softmax()
output = softmax(data)
print("Softmax output:", output)

# Using SoftmaxStable
softmax_stable = SoftmaxStable()
output_stable = softmax_stable(data)
print("SoftmaxStable output:", output_stable)
