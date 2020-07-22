# Muzero-vs-aigym
Muzero vs Cartpole and LunarLander without TPUs

Based on the pseudocode from 'Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model' (arXiv:1911.08265v2) with added comments to (hopefully) make an easy introduction/tutorial for newcomers.

The goal is to make something that runs reasonably on Google Colab or Paperspace, etc. i.e. without needing 1000 TPUs. :)

I've changed the training function so that it does more in parallel and minimises the number of transfers to the GPU.

The top-level jupyter notebooks try to keep long games to a minimum as this is the slow part. So, Cartpole runs many games at the start when they will be short and adds when the the is a noticable improvement in performance. LunarLander just adds a new game after each training session. LunarLander selects games with a bias to the more recent ones. This attempt to keep things quick probably inhibits the convergence and further refinement would help. 
