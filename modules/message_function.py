import torch
from torch import nn
import numpy as np

class MessageFunction(nn.Module):
  """
  Module which computes the message for a given interaction.
  """

  def compute_message(self, raw_messages, source_nodes, timestamps, memory):
    return None


class MLPMessageFunction(MessageFunction):
  def __init__(self, raw_message_dimension, message_dimension):
    super(MLPMessageFunction, self).__init__()

    self.mlp = self.layers = nn.Sequential(
      nn.Linear(raw_message_dimension, raw_message_dimension // 2),
      nn.ReLU(),
      nn.Linear(raw_message_dimension // 2, message_dimension),
    )

  def compute_message(self, raw_messages, source_nodes, timestamps, memory):
    messages = self.mlp(raw_messages)

    return messages


class IdentityMessageFunction(MessageFunction):

  def compute_message(self, raw_messages, source_nodes, timestamps, memory):

    return raw_messages

class NeighborMessageFunction(MessageFunction):
  def __init__(self, raw_message_dimension, message_dimension, neighbor_dimension, neighbor_finder, device, n_neighbors=20, n_layers=1):
    super(NeighborMessageFunction, self).__init__()

    self.neighbor_finder = neighbor_finder
    self.n_neighbors = n_neighbors
    self.neighbor_dimension = neighbor_dimension
    self.activation = nn.ReLU()
    self.device = device

    # W_1 @ raw_message + W_2 @ neighbor_memories
    self.message_linear = nn.Linear(raw_message_dimension, message_dimension)
    self.neighborhood_linear = nn.Linear(neighbor_dimension, message_dimension)
  
  def compute_message(self, raw_messages, source_nodes, timestamps, memory):
    neighbors, _, _ = self.neighbor_finder.get_temporal_neighbor(source_nodes,
      timestamps.cpu(),
      n_neighbors=self.n_neighbors,
      uniform=True)
    neighbor_memories = memory.get_memory(neighbors.flatten())
    neighbor_memories = neighbor_memories.reshape((len(source_nodes), self.n_neighbors, -1))

    neighbor_agg = neighbor_memories.sum(dim=1)

    message_term = self.message_linear(raw_messages)
    neighborhood_term = self.neighborhood_linear(neighbor_agg)
    
    return self.activation(message_term + neighborhood_term)

    # edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
    # edge_deltas = timestamps[:, np.newaxis] - edge_times
    # edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)



def get_message_function(module_type, raw_message_dimension, message_dimension, memory_dimension, neighbor_finder, device, n_neighbors=20, n_layers=1):
  if module_type == "mlp":
    return MLPMessageFunction(raw_message_dimension, message_dimension)
  elif module_type == "neighbor":
    return NeighborMessageFunction(raw_message_dimension, message_dimension, memory_dimension, neighbor_finder, device, n_neighbors=n_neighbors, n_layers=n_layers)
  elif module_type == "identity":
    return IdentityMessageFunction()


## Architecture that runs a GNN on the memory representations of the nodes
## Was too slow/computationally expensive to run locally on my laptop
class MultiLayerNeighborMessageFunction(MessageFunction):
  def __init__(self, raw_message_dimension, message_dimension, neighbor_dimension, neighbor_finder, device, n_neighbors=20, n_layers=1):
    super(MultiLayerNeighborMessageFunction, self).__init__()

    self.neighbor_finder = neighbor_finder
    self.n_neighbors = n_neighbors
    self.n_layers = n_layers
    self.neighbor_dimension = neighbor_dimension
    self.activation = nn.ReLU()
    self.device = device
    # W_1 @ raw_message + W_2 @ neighbor_memories

    self.W_1 = torch.nn.ModuleList([nn.Linear(neighbor_dimension, message_dimension)] + [nn.Linear(message_dimension, message_dimension) for _ in range(n_layers - 1)])

    self.W_2 = torch.nn.ModuleList([nn.Linear(neighbor_dimension, message_dimension)] + [nn.Linear(message_dimension, message_dimension) for _ in range(n_layers - 1)])


    self.message_linear = nn.Linear(raw_message_dimension, message_dimension)
    self.neighborhood_agg_linear = nn.Linear(message_dimension, message_dimension)

  
  # let L be the layer
  # representation(L) = ReLU(A @ messages + B @ neighbor_agg(L))
  # neighbor_agg(L) = sum(neighbor_representations(L-1))
  # neighbor_representations(L) = W_1^L @ neighbor_representations(L-1)

  def neighborhood_agg_rep(self, memory, source_nodes, timestamps, layer):
    neighbors, _, _ = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps.cpu(),
        n_neighbors=self.n_neighbors,
        uniform=True)
    if layer == 0:
      return memory.get_memory(source_nodes)
    else:
      node_representation = self.neighborhood_agg_rep(memory, source_nodes, timestamps, layer - 1)

      neighbors, _, _ = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps.cpu(),
        n_neighbors=self.n_neighbors,
        uniform=True)
      
      neighbors = neighbors.flatten()
      neighbor_representations = self.neighborhood_agg_rep(memory, neighbors, np.repeat(timestamps, self.n_neighbors), layer - 1)
      neighbor_representations = neighbor_representations.reshape((len(source_nodes), self.n_neighbors, -1)).sum(dim=1)

      return self.activation(self.W_1[layer - 1](node_representation) + self.W_2[layer - 1](neighbor_representations))
  

  def compute_message(self, raw_messages, source_nodes, timestamps, memory):
    message_term = self.message_linear(raw_messages)
    neighborhood_agg = self.neighborhood_agg_rep(memory, source_nodes, timestamps, self.n_layers)
    neighborhood_agg = self.neighborhood_agg_linear(neighborhood_agg)

    
    return self.activation(message_term + neighborhood_agg)