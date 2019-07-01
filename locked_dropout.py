import torch
import torch.nn as nn
from torch.autograd import Variable

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout) #why 1-dropout?
        mask = mask.expand_as(x)
        return mask * x

if __name__ == "__main__":
    x = Variable(torch.FloatTensor([[[2,2,3],[4,5,6]],[[2,2,3],[4,5,6]]]))
    drop = LockedDropout()
    print(drop(x))