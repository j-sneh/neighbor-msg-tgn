# Neighbor Aggregated TGN - GRL Mini Project

## Introduction

This is an extension of the original TGN paper, which implements a message function that aggregates over the neighbors representations stored in memory. Running the code is the same as the original TGN, with the following additions:

- Added --message_function "neighbor" as one of possible arguments for the script
- Plotting of T-SNE dimension reduction for each model trial
- Calculations of MAD and measures for node representation convergence/oversmoothing

[See the original TGN Repo](https://github.com/twitter-research/tgn)