#todo: clean citation and commenting.
#This code is basd on https://pytorch-geometric.readthedocs.io/en/2.6.0/_modules/torch_geometric/explain/metric/fidelity.html#fidelity
#get_masked_prediction: https://pytorch-geometric.readthedocs.io/en/2.6.0/_modules/torch_geometric/explain/explainer.html#Explainer.get_masked_prediction

from typing import Tuple
import torch
from torch import Tensor

from torch_geometric.explain import Explainer, Explanation
from torch_geometric.explain.config import ExplanationType, ModelMode


def fidelity_edge_attr(
    explainer: Explainer,
    explanation: Explanation,
    k_percent: float=0.20,
) -> Tuple[float, float]:
    #todo: Update
    r"""Evaluates the fidelity of an
    :class:`~torch_geometric.explain.Explainer` given an
    :class:`~torch_geometric.explain.Explanation`, as described in the
    `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for
    Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper.

    Fidelity evaluates the contribution of the produced explanatory subgraph
    to the initial prediction, either by giving only the subgraph to the model
    (fidelity-) or by removing it from the entire graph (fidelity+).
    The fidelity scores capture how good an explainable model reproduces the
    natural phenomenon or the GNN model logic.

    For **phenomenon** explanations, the fidelity scores are given by:

    .. math::
        \textrm{fid}_{+} &= \frac{1}{N} \sum_{i = 1}^N
        \| \mathbb{1}(\hat{y}_i = y_i) -
        \mathbb{1}( \hat{y}_i^{G_{C \setminus S}} = y_i) \|

        \textrm{fid}_{-} &= \frac{1}{N} \sum_{i = 1}^N
        \| \mathbb{1}(\hat{y}_i = y_i) -
        \mathbb{1}( \hat{y}_i^{G_S} = y_i) \|

    For **model** explanations, the fidelity scores are given by:

    .. math::
        \textrm{fid}_{+} &= 1 - \frac{1}{N} \sum_{i = 1}^N
        \mathbb{1}( \hat{y}_i^{G_{C \setminus S}} = \hat{y}_i)

        \textrm{fid}_{-} &= 1 - \frac{1}{N} \sum_{i = 1}^N
        \mathbb{1}( \hat{y}_i^{G_S} = \hat{y}_i)

    Args:
        explainer (Explainer): The explainer to evaluate.
        explanation (Explanation): The explanation to evaluate.
        k_percent (Flaot): keep/remove top % of pairs
    """

    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

    node_mask = explanation.get('node_mask') #todo: delete?

    #Obtain the mask on edge_feature level
    edge_mask_attr = explanation.get("edge_feat_mask", None)
    if edge_mask_attr is None:
        raise ValueError("edge_feat_mask missing in Explanation")

    kwargs = {key: explanation[key] for key in explanation._model_args}

    # get relevant values
    x         = explanation.x
    edge_ind  = explanation.edge_index
    edge_attr = explanation.edge_attr
    y         = explanation.target
    index     = explanation.get("index")
    model     = explainer.model

    y_hat_all = model(x, edge_ind, edge_attr=edge_attr)

    if explainer.explanation_type == ExplanationType.phenomenon:
        y_hat_all = explainer.get_prediction(
            explanation.x,
            explanation.edge_index,
            **kwargs,
        )
        y_hat_all = explainer.get_target(y_hat)


    #Choose top K% pairs of edge and features.
    E, F = edge_mask_attr.shape
    k_num = max(1, int(k_percent * E * F)) # get the actual number for K

    #Flatten score to use torch.topk(). Will be restored.
    flat_scores = edge_mask_attr.abs().view(-1)

    #get tensor of topk index
    top_idx     = flat_scores.topk(k_num, sorted=False).indices

    #restore
    e_id = top_idx // F   #get row (edge index)  num of feature
    f_id = top_idx %  F   #get column (feature index)

    # Delete selected pairs
    edge_attr_del               = edge_attr.clone()
    edge_attr_del[e_id, f_id]   = 0.
    y_hat_del                   = model(x, edge_ind, edge_attr=edge_attr_del)

    # Use only selected pairs
    edge_attr_add                = torch.zeros_like(edge_attr)  # all zeros
    edge_attr_add[e_id, f_id]    = edge_attr[e_id, f_id]        # restore top-k cells
    y_hat_add                    = model(x, edge_ind, edge_attr=edge_attr_add)

    # convert logits to class
    # prediction is done in logits for Explanation.mode
    if explainer.explanation_type == ExplanationType.model:
        y_hat_all = y_hat_all.argmax(dim=-1)
        y_hat_del = y_hat_del.argmax(dim=-1)
        y_hat_add = y_hat_add.argmax(dim=-1)


    # Indexing In case of phenomeneon to restrict all predictions and lable to that subset.
    # Prediction is already done in class.
    if index is not None:
        y = y[index]
        if explainer.explanation_type == ExplanationType.phenomenon:
            y_hat_all = y_hat_all[index]
        y_hat_del  = y_hat_del[index]
        y_hat_add  = y_hat_add[index]


    #Compute fidelity.
    if explainer.explanation_type == ExplanationType.model:
        # As in the original paper, the scores are against the y_hat_all.
        pos_fidelity = 1. - (y_hat_del == y_hat_all).float().mean()
        neg_fidelity = 1. - (y_hat_add == y_hat_all).float().mean()
    else:
        # As in the original paper, the scores are against the ground truth.
        pos_fidelity = ((y_hat_all == y).float() -
                        (y_hat_del == y).float()).abs().mean()
        neg_fidelity = ((y_hat_all == y).float() -
                        (y_hat_add == y).float()).abs().mean()

    return float(pos_fidelity), float(neg_fidelity)

#todo: below maybe can be deleted as it's all the same with original fidelity.
def characterization_score(
    pos_fidelity: Tensor,
    neg_fidelity: Tensor,
    pos_weight: float = 0.5,
    neg_weight: float = 0.5,
) -> Tensor:
    r"""Returns the componentwise characterization score as described in the
    `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for
    Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper.

    ..  math::
       \textrm{charact} = \frac{w_{+} + w_{-}}{\frac{w_{+}}{\textrm{fid}_{+}} +
        \frac{w_{-}}{1 - \textrm{fid}_{-}}}

    Args:
        pos_fidelity (torch.Tensor): The positive fidelity
            :math:`\textrm{fid}_{+}`.
        neg_fidelity (torch.Tensor): The negative fidelity
            :math:`\textrm{fid}_{-}`.
        pos_weight (float, optional): The weight :math:`w_{+}` for
            :math:`\textrm{fid}_{+}`. (default: :obj:`0.5`)
        neg_weight (float, optional): The weight :math:`w_{-}` for
            :math:`\textrm{fid}_{-}`. (default: :obj:`0.5`)
    """
    if (pos_weight + neg_weight) != 1.0:
        raise ValueError(f"The weights need to sum up to 1 "
                         f"(got {pos_weight} and {neg_weight})")

    denom = (pos_weight / pos_fidelity) + (neg_weight / (1. - neg_fidelity))
    return 1. / denom


def fidelity_curve_auc(
    pos_fidelity: Tensor,
    neg_fidelity: Tensor,
    x: Tensor,
) -> Tensor:
    r"""Returns the AUC for the fidelity curve as described in the
    `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for
    Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper.

    More precisely, returns the AUC of

    .. math::
        f(x) = \frac{\textrm{fid}_{+}}{1 - \textrm{fid}_{-}}

    Args:
        pos_fidelity (torch.Tensor): The positive fidelity
            :math:`\textrm{fid}_{+}`.
        neg_fidelity (torch.Tensor): The negative fidelity
            :math:`\textrm{fid}_{-}`.
        x (torch.Tensor): Tensor containing the points on the :math:`x`-axis.
            Needs to be sorted in ascending order.
    """
    if torch.any(neg_fidelity == 1):
        raise ValueError("There exists negative fidelity values containing 1, "
                         "leading to a division by zero")

    y = pos_fidelity / (1. - neg_fidelity)
    return auc(x, y)


def auc(x: Tensor, y: Tensor) -> Tensor:
    if torch.any(x.diff() < 0):
        raise ValueError("'x' must be given in ascending order")
    return torch.trapezoid(y, x)