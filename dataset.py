import os.path as osp
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
import torch_geometric.transforms as T
import torch


def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Planetoid(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset


def get_amazon_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Amazon(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def get_coauthor_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Coauthor(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_coauthor_amazon_splits(data, num_classes, lcc_mask):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing

    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data