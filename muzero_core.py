from mcts import Node, run_mcts
from gamecomponents import Policy
from game import Game, gameStateMinMax
from replaybuffer import ReplayBuffer
from network import Network, SharedStorage
import random
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional
from helpers import KnownBounds

#
def visit_softmax_temperature(num_moves, training_steps  = 10000, step_limit: int = 10000):
  if training_steps < step_limit:
    return 10.0-9.*training_steps/step_limit
  else:
    return 1.0   


##
 
# Basic configuration object. 
# We break out some of the parameters to be class attributes

class MuZeroConfig(object):

    def __init__(self,
                max_moves: int,
                discount: float,
                num_simulations: int,
                batch_size: int,
                td_steps: int,
                lr_init: float,
                lr_decay_rate: float,
                lr_decay_steps: float,
                visit_softmax_temperature_fn,
                known_bounds: Optional[KnownBounds] = None):

        # game information
        self.max_moves = max_moves
        self.known_bounds = known_bounds

        # MCTS
        self.num_simulations = 100 # number of simulations done by each MCTS step
        self.discount = discount # normal MDP discount 

        # training
        self.training_steps = 10000 # steps to be carried out on each call to network training
        self.checkpoint_interval = 100 # how many training stpes between array checkpoints
        self.num_unroll_steps = 4 # unroll steps to be passed to training
        self.td_steps = td_steps # TD bootstrap steps to be passed to training
        self.weight_decay = 0.00001 
        self.hidden_state_dampen = 0.5 
        # not in the paper, we parameterise the gradient scaling applied to the hidden state

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps

        self.biased_game_sampling = False

        self.visit_softmax_temperature = visit_softmax_temperature_fn
        
    def new_game(self):
        g = Game(self.discount)
        return g

# Create an AI Gym configuration.

def make_aigym_config(name):

    # Temporarily create game to extract useful information
    Game.environment = name
    g = Game(0.9)
    action_list = g.legal_actions()
    state_len = g.image_size[0]

    # normaliser
    gameStateMinMax.__init__([state_len])
    
    # Policy
    Policy.action_list = action_list

    # Network dimensions
    Network.action_count = len(action_list)
    Network.input_size = state_len
    Network.N = 2 # hidden state size
    Network.grad_scale = (1.,1.,1.) # Extra parameter to allow the balance between losses to be adjusted
    Network.input_size = state_len

    # MCTS constants
    Node.root_dirichlet_alpha = 0.3
    Node.root_exploration_fraction = 0.25
    Node.pb_c_base = 19652
    Node.pb_c_init = 1.25

    c = MuZeroConfig(max_moves = 200,
                discount = 1.0,
                num_simulations = 100,
                batch_size = 16,
                td_steps = 50,
                lr_init = 0.002,
                lr_decay_rate = 0.1,
                lr_decay_steps = 50e3,
                visit_softmax_temperature_fn = visit_softmax_temperature)

    return c

##

# Play a game using MCTS
def play_game(config: MuZeroConfig, network: Network) -> Game:

    game = config.new_game()

    while not game.terminal() and len(game.history) < config.max_moves:
        root = Node(0)
        current_observation = game.make_image(-1)

        # fill in the essential information for the root node, using
        # the network output for the current state
        root.expand_node(game.to_play(), game.legal_actions(),
                        network.initial_inference(current_observation)) 
        
        # Exploration is a combination of adding noise to the MCTS process 
        # and selecting the final action with temperature. The latter makes 
        # it more likely that an action/child with low visits is selected; 
        # the former makes it more likely that an action/child will be visited 
        # at least once.
        root.add_exploration_noise()

        # Run the MCTS algorithm starting at root.
        run_mcts(config, root, game.action_history(), network)
        
        # Make the temperature dynamic over a game or training
        T = config.visit_softmax_temperature(num_moves=len(game.history), training_steps = network.training_steps())
        
        action = root.select_action_with_temperature(T) 
        game.apply(action)

        # store in the gmae object all the information we will used for 
        # training
        game.store_search_statistics(root) 

    return game

##

# Train the network
def train_network(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    
    network = storage.latest_network() # recover the latest network to be updated
    
    learning_rate = config.lr_init * config.lr_decay_rate**(network.training_steps()/config.lr_decay_steps)
    
    network.optimiser.learning_rate = learning_rate
    
    for i in range(config.training_steps+1):
        
        if i % config.checkpoint_interval == 0:
            storage.save_network(network.training_steps(), network)

        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps, config.biased_game_sampling) 

        l = network.update_weights(batch, config.weight_decay, config.hidden_state_dampen)

        if i % 100 == 0:
            print((i, l))
            
    storage.save_network(network.training_steps(), network)
    
    return i

##