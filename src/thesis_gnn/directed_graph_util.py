
import copy
import torch
from torch import nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear
from torch import Tensor
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)

class DirGINEWrapper(nn.Module):
    """A wrapper for computing Graph convolution edge-aware conv (e.g. GINEConv) on forward and reverse edges.
    The code is based on DirGNNConv in the torch_geometric.nn.conv.dir_gnn_conv, which does not take edge_feature.

    Args:
    conv : MessagePassing
        The base convolution (must accept (x, edge_index, edge_attr)).
    alpha : float, default=0.5
        Blend factor:  α·incoming + (1-α)·outgoing.
    root_weight : bool, default=True
        If True, adds a learnable self-loop transform (like GIN/GINE).
    """

    def __init__(self, base_conv: GINEConv, alpha: float = 0.5, root_weight: bool = True):
        super().__init__()


        self.alpha = alpha
        self.root_weight = root_weight

        # Make 2 copies for in and out flow of message passing
        self.conv_in = copy.deepcopy(base_conv)
        self.conv_out = copy.deepcopy(base_conv)

        if hasattr(base_conv, 'add_self_loops'):
            self.conv_in.add_self_loops = False
            self.conv_out.add_self_loops = False
        if hasattr(base_conv, 'root_weight'):
            self.conv_in.root_weight = False
            self.conv_out.root_weight = False

        # same rule as DirGNNConv.
        if root_weight:
            self.lin = torch.nn.Linear(base_conv.in_channels, base_conv.out_channels)
        else:
            self.lin = None

        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        self.conv_in.reset_parameters()
        self.conv_out.reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: OptTensor = None) -> Tensor:

        x_in = self.conv_in(x, edge_index, edge_attr)
        x_out = self.conv_out(x, edge_index.flip([0]), edge_attr)

        out = self.alpha * x_out + (1 - self.alpha) * x_in

        if self.root_weight:
            out = out + self.lin(x)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.conv_in}, alpha={self.alpha})'
