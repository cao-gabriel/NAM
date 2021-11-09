import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.spatial.distance as spd
import networkx as nx 
import matplotlib.pyplot as plt 
import mnist
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score

def KNN_graph(data_distances, KNN):
    num_nodes = data_distances.shape[0]
    source_vec, destin_vec, weight_vec = [], [], []
    
    sorted_mtx = np.argsort(data_distances, axis=1)
    for row in range(sorted_mtx.shape[0]):
        for idx in range(1, KNN):
            source_vec.append( row )
            destin_vec.append( sorted_mtx[row, idx] )
            source_vec.append( sorted_mtx[row, idx] )
            destin_vec.append( row )
            weight_vec.append( np.exp(- data_distances[row, sorted_mtx[row, idx]] / 10000  ) )
            weight_vec.append( np.exp(- data_distances[row, sorted_mtx[row, idx]] / 10000  ) )

    Adj = sps.coo_matrix((weight_vec,(source_vec, destin_vec)), shape=(num_nodes, num_nodes)).tocsr()
    Adj = Adj - sps.dia_matrix((Adj.diagonal()[sp.newaxis, :], [0]), shape=(Adj.shape[0], Adj.shape[0]))
    Adj.sum_duplicates()
    return Adj

def get_distances(data_matrix):
    distance_matrix = spd.squareform(spd.pdist(data_matrix))
    return distance_matrix

def get_mnist_KNN_network(mnist_path, KNN=15, images_per_digit=300, seed=0):

    mndata = mnist.MNIST(mnist_path)
    mnist_images, mnist_labels = mndata.load_testing()
    
    mnist_images, mnist_labels, 
    if seed != 0: np.random.seed(seed)

    #### Select a subset of images for each class
    labels_data = np.array(mnist_labels)
    num_classes = np.unique(labels_data).size
    chosen_images_dict = {}
    data_matrix, label_vec = [], []
    for curr_class in range(labels_data.min(), labels_data.max() + 1):
        class_indices = np.where(labels_data == curr_class)[0]
        chosen_indices = np.random.permutation(class_indices)[0:images_per_digit]
        chosen_images_dict[curr_class] = chosen_indices
        data_matrix += [mnist_images[i] for i in chosen_indices]
        label_vec += [curr_class for i in chosen_indices]
    data_matrix = np.vstack(data_matrix)
    label_vec = np.vstack(label_vec).flatten()
    label_dict = {}
    for i in range(label_vec.size):
        label_dict[i] = label_vec[i]

    #### Distance matrix
    data_distances = get_distances(data_matrix)	

    #### KNN Graph
    Adj = KNN_graph(data_distances, KNN)

    return Adj, label_dict

def get_SBM_network(nodes_community, num_communities, deg_in, deg_out, seed=0):
    p_in = deg_in / nodes_community
    p_out = deg_out / nodes_community*(num_communities - 1)
    G = nx.planted_partition_graph(num_communities, nodes_community, p_in, p_out, seed)
    Adj = nx.adjacency_matrix(G).asfptype()
    label_dict = {}
    for i in range(Adj.shape[0]):
        curr_class = i // nodes_community
        label_dict[i] = curr_class
    return Adj, label_dict

def network_visualization(Adj, zoom=30, layout=None, community_assignment={}):
    G = nx.from_scipy_sparse_matrix( Adj )
    if layout==None:
        layout = nx.spring_layout( G ) 
    plt.figure(figsize=(zoom,zoom))
    if len(community_assignment) == 0:
        nx.draw(G, pos=layout)
    else:
        nx.draw(G, pos=layout, node_color=list(community_assignment.values()))
    return layout

def louvain_communities(Adj):
    G = nx.from_scipy_sparse_matrix(Adj)
    return community_louvain.best_partition(G)

def nmi_score(Adj, ground_truth, community_assignment):
    gt = list(ground_truth.values())
    ca = list(community_assignment.values())
    return normalized_mutual_info_score(gt, ca)

def indic_func(idx_list, N):
    indic = np.zeros([N,1])
    indic[idx_list] = 1
    return indic

def modularity_score(Adj, community_assignment):
    # Get a dictionary where keys are labels and values are a list with the nodes having that label
    reverse_assignment = {}
    for node_id in community_assignment.keys(): 
        node_community =  community_assignment[node_id] 
        if node_community not in reverse_assignment.keys():
            reverse_assignment[node_community]=[]
        reverse_assignment[node_community].append( node_id )
        
    # Build community indicator functions
    indic_list = []
    for label in reverse_assignment.keys():
        indic = indic_func( reverse_assignment[label], len(community_assignment))
        indic_list.append( indic )
    
    # Compute modularity
    deg = Adj.sum(axis=1)
    vol_G = np.sum(deg)
    modularity = 0
    for S_i in indic_list:
        
        network_in = S_i.transpose().dot( Adj.dot(S_i) )
        expected_in = deg.transpose().dot(S_i)**2 / vol_G
        modularity += network_in[0,0] - expected_in[0,0]
        
    return modularity / vol_G

from itertools import chain, combinations
def powerset(iterable):
    s = list(iterable)
    return list( chain.from_iterable(combinations(s, r) for r in range(len(s)+1)) )
