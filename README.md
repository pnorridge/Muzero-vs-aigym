# Muzero-vs-aigym
Muzero vs Cartpole without TPUs

Based on the pseudocode from 'Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model' (arXiv:1911.08265v2) with added comments to (hopefully) make an easy introduction to the approach.

The goal is to make something that runs reasonably on Google Colab or Paperspace, etc. i.e. without needing 1000 TPUs. :)

I've changed the training function so that it does more in parallel and minimises the number of transfers to the GPU.

The top-level jupyter notebook tries to keep long games to a minimum as this is the slow part. Probably this inhibits the convergence and further refinement would help. 
