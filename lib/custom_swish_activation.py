
import torch
from torch import nn
from torch.nn import functional as F

class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * i.sigmoid()
        ctx.save_for_backward(result, i)
        return result


    @staticmethod
    def backward(ctx, grad_output):
        result, i = ctx.saved_variables
        sigmoid_x = i.sigmoid()
        return grad_output * (result + sigmoid_x * (1 - result))


swish = Swish.apply


class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)


swish_layer = Swish_module()