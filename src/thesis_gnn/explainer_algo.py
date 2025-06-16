# todo: clean citation and commenting.
# this code is based on the original GNNexplainr algo. https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/explain/algorithm/gnn_explainer.html
from math import sqrt
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain import ExplainerConfig, Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode

class GNNExplainerEdgeFeature(ExplainerAlgorithm):
    # todo: to be updated.
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and node features that play a crucial role in the predictions
    made by a GNN.

    .. note::

        For an example of using :class:`GNNExplainer`, see
        `examples/explain/gnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer.py>`_,
        `examples/explain/gnn_explainer_ba_shapes.py <https://github.com/
        pyg-team/pytorch_geometric/blob/master/examples/
        explain/gnn_explainer_ba_shapes.py>`_, and `examples/explain/
        gnn_explainer_link_pred.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer_link_pred.py>`_.

    .. note::

        The :obj:`edge_size` coefficient is multiplied by the number of nodes
        in the explanation at every iteration, and the resulting value is added
        to the loss as a regularization term, with the goal of producing
        compact explanations.
        A higher value will push the algorithm towards explanations with less
        elements.
        Consider adjusting the :obj:`edge_size` coefficient according to the
        average node degree in the dataset, especially if this value is bigger
        than in the datasets used in the original paper.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.

       Minimal GNN explainer that **ignores node features** but learns an
    edge mask at *structure* or *attribute* granularity.

    ────────────────  SUPPORTED MASK TYPES  ────────────────
    node_mask_type : None                 • no node mask
                     MaskType.object      • one scalar per node (N × 1)

    edge_mask_type : None                 • no edge mask
                     MaskType.object      • one scalar per edge           (E)
                     MaskType.attributes  • one scalar per edge-feature   (E×Fₑ)
                     MaskType.common_attributes • one scalar per feature (1×Fₑ)
    ────────────────────────────────────────────────────────

    Hyper-parameters (can be overridden via kwargs):
        edge_size, edge_reduction, edge_ent,
        node_feat_size, node_feat_reduction, node_feat_ent, EPS,
        epochs, lr
    """

    coeffs = {
        # penalty for structure mask (1-D edge mask)
        "edge_size": 0.005,
        "edge_reduction": "sum",  # todo: is mean better than sum? #todo: not too sure what this is.
        "edge_ent": 1.0,

        # penalty for attribute-level edge mask
        "edge_feat_size": 1.0,
        "edge_feat_reduction": "mean",
        "edge_feat_ent": 0.1,

        # penalty for node-mask
        "node_feat_size": 1.0,  # used only if node mask active
        "node_feat_reduction": "mean",  # todo: is mean better than sum?
        "node_feat_ent": 0.1,

        "EPS": 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None

    def forward(
            self,
            model: torch.nn.Module,
            x: Optional[Tensor],
            edge_index: Tensor,
            edge_attr: Tensor,
            *,
            target: Tensor,
            index: Optional[Union[int, Tensor]] = None,
            **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        self._train(model, x, edge_index, edge_attr, target=target, index=index, **kwargs)

        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask,
            apply_sigmoid=True,
        )
        edge_mask = self._post_process_mask(
            self.edge_mask,
            self.hard_edge_mask,
            apply_sigmoid=True,
        )

        edge_mask_proxy = edge_mask.mean(dim=-1) # 1D tensor to bypass validators

        self._clean_model(model)

        # return explanation with relevant items together. As this algo might get called directly.
        return Explanation(node_mask=node_mask, # (N,1) or None
                        edge_mask      = edge_mask_proxy,    # (E,)   to bypass validator
                        edge_feat_mask = edge_mask,          # (E,F)  mask with full attribute
                        edge_attr      = edge_attr,
                        x              = x,
                        edge_index     = edge_index,
                        target         = target,
            )

    def supports(self) -> bool:
        return True

    def _train(
            self,
            model: torch.nn.Module,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Tensor,
            *,
            target: Tensor,
            index: Optional[Union[int, Tensor]] = None,
            **kwargs,
    ):
        self._initialize_masks(x, edge_index, edge_attr)

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)

        if self.edge_mask is not None:
            # 1-D structure mask must be attached.
            if self.edge_mask.ndim == 1:  # if it's for structure mask.
                set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in range(self.epochs):
            optimizer.zero_grad()

            h = x if self.node_mask is None else x * self.node_mask.sigmoid()

            if self.edge_mask is not None and self.edge_mask.ndim >= 2:
                masked_attr = edge_attr * self.edge_mask.sigmoid()
            else:
                masked_attr = edge_attr

            y_hat, y = model(h, edge_index, edge_attr=masked_attr, **kwargs), target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)

            loss.backward()
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).
            if i == 0 and self.node_mask is not None:
                if self.node_mask.grad is None:
                    raise ValueError("Could not compute gradients for node "
                                     "features. Please make sure that node "
                                     "features are used inside the model or "
                                     "disable it via `node_mask_type=None`.")
                self.hard_node_mask = self.node_mask.grad != 0.0
            if i == 0 and self.edge_mask is not None:
                if self.edge_mask.grad is None:
                    raise ValueError("Could not compute gradients for edges. "
                                     "Please make sure that edges are used "
                                     "via message passing inside the model or "
                                     "disable it via `edge_mask_type=None`.")
                self.hard_edge_mask = self.edge_mask.grad != 0.0

    def _initialize_masks(self, x: Optional[Tensor], edge_index: Tensor, edge_attr: Tensor):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        device = x.device
        std = 0.1

        # Node mask (only "object" is supported)
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            N = x.size(0)
            self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
        else:
            assert False

        # Edge mask
        E, F_e = edge_attr.size()

        if edge_mask_type is None:
            self.edge_mask = None

        elif edge_mask_type == MaskType.object:
            # std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            # self.edge_mask = Parameter(torch.randn(E, device=device) * std)
            # todo: double chekc which one is better
            # use E instead of N
            gain = torch.nn.init.calculate_gain("relu")
            self.edge_mask = Parameter(
                torch.randn(E, device=device)
                * gain
                * sqrt(2.0 / (2 * max(E, 1)))
            )


        elif edge_mask_type == MaskType.attributes:
            self.edge_mask = Parameter(torch.randn(E, F_e, device=device) * std)

        elif edge_mask_type == MaskType.common_attributes:
            self.edge_mask = Parameter(torch.randn(1, F_e, device=device) * std)

        else:
            assert False

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            assert False

        # penalty for egde structure mask
        if self.hard_edge_mask is not None and self.edge_mask.ndim == 1:
            assert self.edge_mask is not None
            m = self.edge_mask[self.hard_edge_mask].sigmoid()
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                    1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

        # penalty for edge attribute level
        if self.hard_edge_mask is not None and self.edge_mask.ndim >= 2:
            assert self.edge_mask is not None
            m = self.edge_mask[self.hard_edge_mask].sigmoid()
            edge_reduce = getattr(torch, self.coeffs["edge_feat_reduction"])
            loss = loss + self.coeffs["edge_feat_size"] * edge_reduce(m)
            ent = -m * torch.log(m + self.coeffs["EPS"]) - (
                    1 - m) * torch.log(1 - m + self.coeffs["EPS"])
            loss = loss + self.coeffs["edge_feat_ent"] * ent.mean()

        if self.hard_node_mask is not None:
            assert self.node_mask is not None
            m = self.node_mask[self.hard_node_mask].sigmoid()
            node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
            loss = loss + self.coeffs['node_feat_size'] * node_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                    1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None


# function to force overwriting the mask_type attribute to pass the error message.
def make_attr_explainer(algorithm, mdl_cfg):
    cfg = ExplainerConfig("model", node_mask_type="object", edge_mask_type="object")
    algorithm.connect(explainer_config=cfg, model_config=mdl_cfg)
    algorithm._explainer_config.edge_mask_type = MaskType.attributes
    return algorithm

def make_common_attr_explainer(algorithm, mdl_cfg):
    cfg = ExplainerConfig("model", node_mask_type="object", edge_mask_type="object")
    algorithm.connect(explainer_config=cfg, model_config=mdl_cfg)
    algorithm._explainer_config.edge_mask_type = MaskType.common_attributes
    return algorithm

