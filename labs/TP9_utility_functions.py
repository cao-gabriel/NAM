import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.manifold import TSNE

from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.transforms import NormalizeFeatures
from neural_network_models import *

def compute_loss(criterion, out, gt, mask):
    return criterion(out[mask], gt[mask])
    
def compute_accuracy(out, gt, mask):
    pred = out.argmax(dim=1)
    train_correct = pred[mask] == gt[mask]
    return int(train_correct.sum()) / int(mask.sum())

def loss_display(loss_train, acc_train, loss_val, acc_val, epoch, num_epochs, use_val):
    clear_output(wait=True)
            
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot( range(0, epoch + 1), loss_train, label='Train loss')
    if use_val == True:
        plt.plot( range(0, epoch + 1), loss_val, label='Val loss')
        
    plt.xlim([0,num_epochs])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot( range(0, epoch + 1), acc_train, label='Train accuracy')
    if use_val == True:
        plt.plot( range(0, epoch + 1), acc_val, label='Val accuracy')    
    plt.xlim([0,num_epochs])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def visualize(h, true_color, pred_color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=pred_color, cmap="Set2")
    plt.title('Predicted class')
    plt.subplot(1,2,2)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=true_color, cmap="Set2")
    plt.title('True class')
    plt.show()
    
def load_cora_dataset(num_train_per_class, num_val, num_test, seed=0):
    torch.manual_seed(seed)
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures(), split='random', \
                       num_train_per_class=num_train_per_class, num_val=num_val, num_test=num_test)
    return dataset

def load_protein_dataset(num_train_per_class, num_val, num_test, batch_size, seed=0):
    torch.manual_seed(seed)
    dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
    
    graph_class = torch.tensor([graph.y.item() for graph in dataset])
    zero_idx = torch.where(graph_class == 0)[0]
    ones_idx = torch.where(graph_class == 1)[0]
    
    zero_idx_perm = zero_idx[ torch.randperm(len(zero_idx)) ]
    ones_idx_perm = ones_idx[ torch.randperm(len(ones_idx)) ]
    
    train_mask_c1 = zero_idx_perm[0:num_train_per_class]
    train_mask_c2 = ones_idx_perm[0:num_train_per_class]
    train_mask = torch.cat( (train_mask_c1, train_mask_c2), dim=0 )
    
    val_mask_c1 =  zero_idx_perm[num_train_per_class: num_train_per_class + math.ceil(num_val/2)]
    val_mask_c2 = ones_idx_perm[num_train_per_class: num_train_per_class + math.floor(num_val/2)]
    val_mask = torch.cat( (val_mask_c1, val_mask_c2), dim=0 )
    
    test_mask_c1 = zero_idx_perm[len(zero_idx_perm) - math.ceil(num_test/2):]
    test_mask_c2 = ones_idx_perm[len(ones_idx_perm) - math.floor(num_test/2):]
    test_mask = torch.cat( (test_mask_c1, test_mask_c2), dim=0 )
    
    dataset.train_loader = DataLoader( dataset[train_mask], batch_size=batch_size, shuffle=True)
    dataset.val_loader = DataLoader( dataset[val_mask], batch_size=len(val_mask) )
    dataset.test_loader = DataLoader( dataset[test_mask], batch_size=len(test_mask) )
       
    return dataset

################################################################
#################### Multi-layer perceptron ####################
################################################################

def get_MLP_1_hidden(in_dimension, hid_dimension, out_dimension):
    model = MLP_1_hidden_model(in_dimension, hid_dimension, out_dimension)
    return model

def train_MLP_1_hidden(model, data, num_epochs, lr=0.01, use_val=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    loss_train_list, loss_val_list = [], []
    acc_train_list, acc_val_list = [], []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x)
        
        ## Loss and accuracy
        loss_train = compute_loss(criterion, out, data.y, data.train_mask)
        loss_val = compute_loss(criterion, out, data.y, data.val_mask)
        
        acc_train = compute_accuracy(out, data.y, data.train_mask)
        acc_val = compute_accuracy(out, data.y, data.val_mask)
        
        ## Backpropagate
        loss_train.backward()
        optimizer.step()
        
        ## Update lists for plotting
        loss_train_list += [ loss_train.item() ]
        loss_val_list += [ loss_val.item() ]
        
        acc_train_list += [ acc_train ]
        acc_val_list += [ acc_val ]
        
        if epoch % 10 == 0:
            loss_display( loss_train_list, acc_train_list, loss_val_list, acc_val_list, epoch, num_epochs, use_val )
             
    return model

def test_MLP_1_hidden(model, data):
    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    visualize(out, true_color=data.y, pred_color=pred)
    return test_acc

##############################################################
#################### GCN - 1 Hidden layer ####################
##############################################################

def get_GCN_1_hidden(in_dimension, hid_dimension, out_dimension):
    model = GCN_1_hidden_model(in_dimension, hid_dimension, out_dimension)
    return model

def train_GCN_1_hidden(model, data, num_epochs, lr=0.01, use_val=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    loss_train_list, loss_val_list = [], []
    acc_train_list, acc_val_list = [], []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        ## Loss and accuracy
        loss_train = compute_loss(criterion, out, data.y, data.train_mask)
        loss_val = compute_loss(criterion, out, data.y, data.val_mask)
        
        acc_train = compute_accuracy(out, data.y, data.train_mask)
        acc_val = compute_accuracy(out, data.y, data.val_mask)
        
        ## Backpropagate
        loss_train.backward()
        optimizer.step()
        
        ## Update lists for plotting
        loss_train_list += [ loss_train.item() ]
        loss_val_list += [ loss_val.item() ]
        
        acc_train_list += [ acc_train ]
        acc_val_list += [ acc_val ]
        
        if epoch % 10 == 0:
            loss_display( loss_train_list, acc_train_list, loss_val_list, acc_val_list, epoch, num_epochs, use_val )
             
    return model

def test_GCN_1_hidden(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    visualize(out, true_color=data.y, pred_color=pred)
    return test_acc


##############################################################
#################### GCN - 2 Hidden layer ####################
##############################################################

def get_GCN_2_hidden(in_dimension, hid_1_dimension, hid_2_dimension, out_dimension):
    model = GCN_2_hidden_model(in_dimension, hid_1_dimension, hid_2_dimension, out_dimension)
    return model

def train_GCN_2_hidden(model, data, num_epochs, lr=0.01, use_val=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    loss_train_list, loss_val_list = [], []
    acc_train_list, acc_val_list = [], []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        ## Loss and accuracy
        loss_train = compute_loss(criterion, out, data.y, data.train_mask)
        loss_val = compute_loss(criterion, out, data.y, data.val_mask)
        
        acc_train = compute_accuracy(out, data.y, data.train_mask)
        acc_val = compute_accuracy(out, data.y, data.val_mask)
        
        ## Backpropagate
        loss_train.backward()
        optimizer.step()
        
        ## Update lists for plotting
        loss_train_list += [ loss_train.item() ]
        loss_val_list += [ loss_val.item() ]
        
        acc_train_list += [ acc_train ]
        acc_val_list += [ acc_val ]
        
        if epoch % 10 == 0:
            loss_display( loss_train_list, acc_train_list, loss_val_list, acc_val_list, epoch, num_epochs, use_val )
             
    return model

def test_GCN_2_hidden(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    visualize(out, true_color=data.y, pred_color=pred)
    return test_acc


##############################################################
#################### GRAPH CLASSIFICATION ####################
##############################################################

def get_GCN_graph_classif(in_dimension, hid_1_dimension, hid_2_dimension, hid_3_dimension, out_dimension):
    model = GCN_graph_classif_model(in_dimension, hid_1_dimension, hid_2_dimension, hid_3_dimension, out_dimension)
    return model

def batch_train(optimizer, criterion, model, train_loader):
    model.train()
    loss_list = []
    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss_train = criterion(out, data.y)
        loss_train.backward()
        loss_list += [loss_train.item()]
        optimizer.step()
        optimizer.zero_grad()

    return model, sum(loss_list)/len(loss_list)

def batch_eval(model, data_loader):
    model.eval()
    correct = 0
    for data in data_loader: 
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1) 
        correct += int((pred == data.y).sum()) 
    return correct / len(data_loader.dataset)  

def train_GCN_graph_classif(model, dataset, num_epochs, lr=0.01, use_val=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_train_list, loss_val_list = [], []
    acc_train_list, acc_val_list = [], []
    
    for epoch in range(num_epochs):
        model, loss_train = batch_train(optimizer, criterion, model, dataset.train_loader)
        acc_train = batch_eval(model, dataset.train_loader)
        
        # Store result
        loss_train_list += [loss_train]
        acc_train_list += [acc_train]
        
        if use_val == True:
            # Val loss
            loss_list = []
            for data in dataset.val_loader:
                out = model(data.x, data.edge_index, data.batch)
                loss_val = criterion(out, data.y)
                loss_list += [loss_val.item()]
            # Val acc
            acc_val = batch_eval(model, dataset.val_loader)
            
            # Store result
            loss_val_list += [sum(loss_list)/len(loss_list)]
            acc_val_list += [acc_val]
       
        if epoch % 5 == 0:
            loss_display( loss_train_list, acc_train_list, loss_val_list, acc_val_list, epoch, num_epochs, use_val )
            
    return model

def test_GCN_graph_classif(model, dataset):
    model.eval()
    test_acc = batch_eval(model, dataset.test_loader)
    for data in dataset.test_loader:
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1) 
        visualize(out, true_color=data.y, pred_color=pred)
    return test_acc