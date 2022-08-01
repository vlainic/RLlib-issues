# REPORTED issue: https://discuss.ray.io/t/multi-agent-where-does-the-first-structure-comes-from/7010

import numpy as np
import pandas as pd

import gym
from gym import spaces

import networkx as nx

# multi-agent:
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from collections import OrderedDict
from ray.rllib.utils.spaces import repeated
    
# TO BE AUTOMATIZED:
action_maps = {1:1, 2:2, 3:3}
mat_in_only = [101]
mat_out = [102, 103, 104]

df_dict = [
    {'material_focus': 101, 'material_in': 101, 'material_out': 102, 'procnr': 1, 'bom': 1},
    {'material_focus': 101, 'material_in': 101, 'material_out': 102, 'procnr': 2, 'bom': 1},
    {'material_focus': 101, 'material_in': 101, 'material_out': 103, 'procnr': 2, 'bom': 1},
    {'material_focus': 101, 'material_in': 101, 'material_out': 103, 'procnr': 3, 'bom': 1},
    {'material_focus': 101, 'material_in': 101, 'material_out': 104, 'procnr': 3, 'bom': 1},
]
df = pd.DataFrame(df_dict)

def processing_list(decisions, df):
    actions = np.concatenate(([1.0], decisions))
    
    materials = list(df.material_in.unique())
    materials += [x for x in df.material_out.unique() if x not in materials]
    
    cuttings = list(df.procnr.unique())
    
    g = nx.DiGraph()
    g.add_nodes_from(materials, n=0, a=1.0)
    for cc in cuttings:
        g.add_node(cc, a=0.0)
    for ii,row in df.iterrows():
        g.add_edge(row.material_in, row.procnr)
        g.add_edge(row.procnr, row.material_out)
        
    predecessors = {}
    successors = {}
    for edge_in,edge_out in g.edges():
        predecessors.setdefault(edge_out, [])
        predecessors[edge_out].append(edge_in)

        successors.setdefault(edge_in, [])
        successors[edge_in].append(edge_out)
        
    for e_in,e_out in g.edges():
        if e_in in cuttings:
            g.nodes[e_in]['a'] = actions[action_maps[e_in]]
            
    for node in g.nodes():
        if node in df.material_out.unique():
            cut_input = sum([g.nodes[x]['a']*g.nodes[predecessors[x][0]]['a'] for x in predecessors[node]])
            g.nodes[node]['a'] = cut_input
            
    for edge_in,list_out in successors.items():
        if edge_in in materials:
            leftover = 1-sum([g.nodes[x]['a'] for x in list_out])
            g.nodes[edge_in]['a'] *= leftover
            
    return {x:g.nodes[x]['a'] for x in df.material_out.unique()}

def env0_graph_lin():  
        g = nx.DiGraph()
        
        # Main supplier nodes...
        g.add_node("w", stock={x:0.0 for x in mat_in_only})
        
        # Plants
        g.add_node("x", stock={x:0.0 for x in mat_in_only},
                   hold_cost = 0.01, make_cost = 0.05)
        
        # Material nodes
        g.add_node("y", stock={x:0.0 for x in mat_out},
                   hold_cost = 0.02, make_cost = 0.2)
        
        # Markets
        g.add_node("z", stock={x:0.0 for x in mat_out},
                   penalty = 0.5)
        
        # Edges are empty for now
        g.add_edges_from([("w", "x"), ("x", "y"), ("y", "z")])
        
        return g

agent_names = ['x', 'y']
    
class multiagent_public(MultiAgentEnv):
    """
    A simple environment that has 1 farm, 1 slaughter, 3 cutts, 3 packs and 1 market, w/o wasting.
    :param config: Dict that containts following keys:
                   - demand_predicted: Nortura's prediction
                   - demand_realized: what actually happened
                   - supply: based on the predicted demand
                   - prod_price: product prices
                   - backlog_penalty: percentage of price as penalty for not fulfilling the demand
                   - future_steps: how many future steps to use as input, i.e. state
    """

    def __init__(self, config):
        self._skip_env_checking = True # True
        """
        We've added a module for checking environments that are used in experiments. It will cause your environment to fail if your 
        environment is not set upcorrectly. You can disable check env by setting `disable_env_checking` to True in your experiment config 
        dictionary. You can run the environment checking module standalone by calling ray.rllib.utils.check_env(env).
        Skipping env checking for this experiment.
        """
        
        # INPUTS
        self.config = config.copy()
        
        self.demand_pred = config['demand_predicted'].copy()
        self.demand_real = config['demand_realized'].copy()
        self.supply = config['supply'].copy()
        self.prod_price = config['prod_price'].copy()
        self.future_steps = config['future_steps']
        
        self.action_idx_map = {
            'x': [0],
            'y': [1,2,3],
        }
        
        # some constants
        self.n_steps = len(list(self.demand_pred.values())[0]) - self.future_steps # (2*self.future_steps+1)
        self.n_prods = len(self.prod_price)
        
        # THE GRAPH
        self.graph = env0_graph_lin()
        
        # ENVIRONMENT PARAMETERS
#         self._agent_ids = set(["cut_Haerland", "plant_Haerland"])
        self._agent_ids = set(agent_names)
        
        # action space
        self.action_space = spaces.Dict({
            agent_names[0]: spaces.Box(low = np.array([-0.1]), high = np.array([0.1]), dtype = np.float32),
            agent_names[1]: spaces.Box(low = np.array([0.0]*self.n_prods), high = np.array([1.0]*self.n_prods), dtype = np.float32),
        })        
        
        self.step_count = self.future_steps+1
        # state
        self.state, self.n_states = self._get_state(return_len=True)
        
        obs = spaces.Box(low = -10_000, high = 200_000, shape = (self.n_states,), dtype = np.int32)
        obs.seed(0)
        self.observation_space = spaces.Dict({
            agent_names[0]: obs, 
            agent_names[1]: obs, 
        })
        
    def _get_one_state(self):
        
        # current and future supply
        supplies = self.supply[self.step_count:self.step_count+self.future_steps+1].reshape(-1,).astype(np.int32)
        
        # current and future predicted demand
        d_pred = np.array([list(v[self.step_count:self.step_count+self.future_steps+1])
                           for v in self.demand_pred.values()]).reshape(-1).astype(np.int32)
        
        # previous/recent realized demand
        d_past = np.array([list(v[self.step_count-(self.future_steps+1):self.step_count])
                           for v in self.demand_real.values()]).reshape(-1).astype(np.int32)
        
        # inventory/stock
        inv_per_stock = []
        for _,data in self.graph.nodes(data=True):
            inv_per_stock += list(data['stock'].values())
        inv_per_stock = np.array(inv_per_stock).astype(np.int32)
        
        return np.concatenate((supplies, d_pred, d_past, inv_per_stock), dtype=np.int32)
    
    def _get_state(self, return_len=False):
        curr_state = self._get_one_state()
        dict_state = { # OrderedDict(
                agent_names[0]: curr_state,
                agent_names[1]: curr_state,
            }#)
        
        if return_len:
            return dict_state, len(curr_state)
        else:
            return dict_state
        
    def step(self, actions):
        '''
        Execute one time step within the environment: i.e. propagate material from supply towards market.
        '''
        
        reward_track = 0.0
        for node_from,node_to in self.graph.edges():
                
            # updating supply with new planned input
            if node_from.startswith('w'):
                self.graph.nodes[node_from]['stock'][101] += self.supply[self.step_count]

                edge_production = {k:v for k,v in zip(self.graph.nodes[node_to]['stock'],
                                                      1+actions[node_to])}

                for product,stock in self.graph.nodes[node_to]['stock'].items():
                    # NOTE: 101 is hardcoded here, later it should be focus material!!!
                    newly_produced = np.round(max(self.graph.nodes[node_from]['stock'][101]*edge_production[product], 0.0))
                    self.graph.nodes[node_to]['stock'][product] = stock + newly_produced

                    reward_track -= newly_produced * self.graph.nodes[node_to]['make_cost']

                # removing consumed material -> again 101 hardcoded!!!
                self.graph.nodes[node_from]['stock'][101] -= newly_produced

            elif node_to.startswith('y'):           
                # cutting list production
                edge_production = processing_list(actions[node_to], df)

                # performing production over input
                for product,stock in self.graph.nodes[node_to]['stock'].items():
                    # NOTE: 101 is hardcoded here, later should be focus material!!!
                    newly_produced = np.round(self.graph.nodes[node_from]['stock'][101]*edge_production[product])
                    self.graph.nodes[node_to]['stock'][product] = stock + newly_produced

                    reward_track -= newly_produced * self.graph.nodes[node_to]['make_cost']
                
                self.graph.nodes[node_from]['stock'][101] = 0.0
                
            else:
                for product in self.graph.nodes[node_to]['stock']:
                    # true demand
                    demand = self.demand_real[product][self.step_count]
                    # what can be delivered
                    delivery = min(self.graph.nodes[node_from]['stock'][product], demand)
                    
                    self.graph.nodes[node_from]['stock'][product] -= delivery
                    reward_track += delivery * self.prod_price[product]
                    # if demand is not fullfilled
                    if delivery < demand:
                        backlog = demand - delivery
                        self.graph.nodes[node_to]['stock'][product] += backlog
                        
                        reward_track -= backlog * self.prod_price[product] * self.graph.nodes[node_to]['penalty']
            
        # finalizing reward: adding the stock cost
        inventory_cost = sum([sum(data['stock'].values())*data['hold_cost']
                              for _,data in self.graph.nodes(data=True) if 'hold_cost' in data])                
        rew = (reward_track - inventory_cost) / 10_000
        rewards = {k:rew for k in self._agent_ids}
                    
        # is eposide done?
        dones = {k:False for k in self._agent_ids}
        dones.update({'__all__':False})
        self.step_count += 1
        if self.step_count % self.n_steps == 0:
            dones['__all__'] = True
        else:            
            self.state = self._get_state()
        
        return self.state, rewards, dones, {'graph': self.graph}

    def reset(self):
        '''Reset the state of the environment to an initial state.'''
        
        # THE GRAPH reset
        self.graph = env0_graph_lin()
            
        self.step_count = self.future_steps+1
        # state
        self.state = self._get_state()
        
        return self.state