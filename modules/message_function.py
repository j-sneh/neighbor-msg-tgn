import torch
from torch import nn

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
    self.device = device
    # W_1 @ raw_message + W_2 @ neighbor_memories
    self.message_sum = self.layers = nn.Sequential(
      nn.Linear(raw_message_dimension, raw_message_dimension // 2),
      nn.ReLU(),
      nn.Linear(raw_message_dimension // 2, message_dimension),
      nn.ReLU(),
    )

    self.neighbor_sum = self.layers = nn.Sequential(
      nn.Linear(neighbor_dimension, message_dimension),
      nn.ReLU()
    )

    # print(f"neighbor_dimension: {neighbor_dimension}, message_dimension: {message_dimension}, raw_message_dimension: {raw_message_dimension}")
    

  def compute_message(self, raw_messages, source_nodes, timestamps, memory):
    neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps.cpu(),
        n_neighbors=self.n_neighbors,
        uniform=True)
    

    # print("neighbors shape: ", neighbors.shape)
    # print("neighbors: ", neighbors)
    # print("raw_messages shape: ", raw_messages.shape)

    neighbors = neighbors.flatten()
    neighbor_memories = memory.get_memory(neighbors)
    neighbor_memories = neighbor_memories.reshape((len(source_nodes), self.n_neighbors, -1))

    # print("neighbor_memories shape: ", neighbor_memories.shape)
    neighbor_agg = neighbor_memories.mean(dim=1)
    # print("neighbor_agg shape: ", neighbor_agg.shape)



    # neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
    a = self.message_sum(raw_messages)
    b = self.neighbor_sum(neighbor_agg)
    messages = a + b

    return messages

    # edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
    # edge_deltas = timestamps[:, np.newaxis] - edge_times
    # edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)



def get_message_function(module_type, raw_message_dimension, message_dimension, memory_dimension, neighbor_finder, device, n_neighbors=20):
  if module_type == "mlp":
    return MLPMessageFunction(raw_message_dimension, message_dimension)
  elif module_type == "neighbor":
    return NeighborMessageFunction(raw_message_dimension, message_dimension, memory_dimension, neighbor_finder, device, n_neighbors=n_neighbors, n_layers=1)
  elif module_type == "identity":
    return IdentityMessageFunction()

