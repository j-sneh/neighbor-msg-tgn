# Neighbor Aggregated TGN - GRL Mini Project

## Introduction

This is an extension (fork) of the original TGN paper, which implements a message function that aggregates over the neighbors representations stored in memory. Running the code is the same as the original TGN, with the following additions:

- Added --message_function "neighbor" as one of possible arguments for the script
- Plotting of T-SNE dimension reduction for each model trial
- Calculations of Dirichlet energy for node representation convergence

[See the original TGN Repo](https://github.com/twitter-research/tgn)

## Running Experiments

For Experiment 1 (Full tgn-attn model with added neighbor message aggregation), run
```
python train_self_supervised.py --use_memory --prefix tgn-attn-neighbormsg --n_runs 5 --message_function neighbor --n_epoch 20
```

For Experiment 2 (tgn-id model with added neighbor message aggregation), run
```
python train_self_supervised.py --use_memory --prefix tgn-id-neighbormsg --n_runs 5 --message_function neighbor --embedding_module identity --n_epoch 20
```

For Experiment 3 (tgn-attn model) *baseline*, run
```
python train_self_supervised.py --use_memory --prefix tgn-baseline --n_runs 5 --message_function identity --n_epoch 20
```

For Experiment 4 (tgn-id model), run
```
python train_self_supervised.py --use_memory --prefix tgn-id --n_runs 5 --embedding_module identity --message_function identity --n_epoch 20
```

## Specific Changed Files

See the updated `train_self_supervised.py`, `modules/message_function.py`, `model/tgn.py`, and `utils/bipartite.py`.