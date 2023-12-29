
import torch
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class Prop(MessagePassing):
    def __init__(self, num_classes, K, bias=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = Linear(num_classes, 1)
        self.weight = Parameter(torch.Tensor(1,self.K + 1))

    def forward(self, x, edge_index, edge_weight=None):
        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)
        out = []
        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)

        pps = torch.stack(preds, dim=1)
        feature_pool = self.proj(pps).squeeze()
        print(feature_pool.shape)
        m_result_w = feature_pool.mean(dim=0)
        share_result_w = F.normalize(m_result_w + self.weight, dim=-1)
        self_result_w = F.normalize(feature_pool.unsqueeze(1), dim=-1)
        share_result = torch.matmul(share_result_w, pps).squeeze()
        self_result = torch.matmul(self_result_w, pps).squeeze()
        # out.append(share_result*0.7
        # out.append(self_result*0.)
        # out_result = torch.stack(out, dim = -1)
        # # print(out_result.shape)
        # out_result = self.rusult(out_result).squeeze()
        out_result = share_result * 0.2 + self_result * 0.8

        return out_result


    # cora 0.8 0.2
    # citeseet 0.4 0.6
    # pubmed 0.2 0.8

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)

    def reset_parameters(self):
        self.proj.reset_parameters()
        glorot(self.weight)


class Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden,bias=True)
        self.prop = Prop(args.hidden, args.K)
        self.lin2 = Linear(args.hidden, dataset.num_classes,bias=True)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data, args):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p = 0.6, training=self.training)
        x = F.selu(self.lin1(x))
        x = F.dropout(x, p = 0.3 , training=self.training)
        x = torch.tanh(self.prop(x, edge_index))
        x = F.dropout(x, p = 0.2, training=self.training)
        x = self.lin2(x)
        # cora 0.55 0.25 0.5 relu tanh / l=0.01 w = 0.002
        # pubmed 0.6 0.3 0.2 selu tanh / l=0.01 w = 0.005
        # citeseer 0.4 0.4 0.4 elu tanh / l=0.02 w=0.008
        # cs 0.25 0.75 0.25 / selu /
        return F.log_softmax(x, dim=1)