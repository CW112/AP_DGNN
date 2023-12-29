import torch
import time
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from torch import tensor
from logger import *
import networkx as nx
from torch_geometric.utils import *

def train(model, optimizer, data,args):
    model.train()
    optimizer.zero_grad()
    out = model(data,args)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

def test_eval(model, data,args):
    model.eval()
    with torch.no_grad():
        logits = model(data,args)
    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc
    return outs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,args,permute_masks,lcc):
    val_losses, accs, durations = [], [], []

    data = dataset[0]
    pbar = tqdm(range(runs), unit='run')

    loss_list = []
    val_acc_list = []
    test_acc_list = []
    acc_lists_str = []
    str_lists = []
    if lcc:  # select largest connected component
        data_ori = dataset[0]
        data_nx = to_networkx(data_ori)
        data_nx = data_nx.to_undirected()
        print("Original #nodes:", data_nx.number_of_nodes())
        data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
        print("#Nodes after lcc:", data_nx.number_of_nodes())
        lcc_mask = list(data_nx.nodes)

    for run_time in pbar:

        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes, None)


        val_acc_list.append("\n " + str(run_time) + " ")
        test_acc_list.append("\n " + str(run_time) + " ")
        loss_list.append("\n " + str(run_time) + " ")
        data = data.to(device)
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        best_val_loss = float('inf')
        val_loss_history = []

        for epoch in range(1, epochs + 1):
            train(model, optimizer, data,args)
            eval_info = test_eval(model, data,args)
            loss_list.append(str( eval_info['train_loss']) + " ")
            val_acc_list.append(str(eval_info['val_acc']) + " ")
            test_acc_list.append(str(eval_info['test_acc']) + " ")

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                test_acc = eval_info['test_acc']
                torch.save({'model_state_dict': model.state_dict()}, "DGNNV2+"+args.dataset+"+"+str(args.K) + ".pt")

            val_loss_history.append(eval_info['val_loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        checkpoint = torch.load( "DGNNV2+"+args.dataset+"+"+str(args.K) + ".pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        eval_info = test_eval(model, data, args)
        val_loss = eval_info['val_loss']
        test_acc = eval_info['test_acc']
        val_losses.append(val_loss)
        accs.append(test_acc)
        print("test_acc:", test_acc)
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()))

    str_lists.append("\n Total acc:")
    str_lists.extend(str(accs))
    str_lists.append("\n mean:{:.4f}".format(acc.mean().item()))
    str_lists.append("\n std:{:.4f}".format(acc.std().item()))
    str_lists.append("\n mean_time:{:.4f}".format(duration.mean().item()))
    str_lists.append("\n std_time:{:.4f}".format(duration.std().item()))
    log_name = "DGNNV2+"+args.dataset +"+"+ str(args.K) + ".txt"
    logging(log_name, str_lists)
    log_loss_name ="DGNNV2+"+ args.dataset +"+"+ str(args.K) +"+loss.txt"
    logging(log_loss_name, loss_list)
    log_val_name ="DGNNV2+"+ args.dataset +"+"+str(args.K) +"+val_acc.txt"
    logging(log_val_name, val_acc_list)
    log_test_name ="DGNNV2+"+ args.dataset +"+"+str(args.K) +"+test_acc.txt"
    logging(log_test_name, test_acc_list)
    val_acc_list.clear()
    test_acc_list.clear()
    acc_lists_str.clear()
    str_lists.clear()
    durations.clear()
    loss_list.clear()
