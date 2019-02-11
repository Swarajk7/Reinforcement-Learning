from env import Easy21Env
from monte_carlo_control import MonteCarloGLIEControl
from plot_graphs import plot_value_func
import numpy as np

if __name__ == '__main__':
    env = Easy21Env()
    
    mc_glie = MonteCarloGLIEControl(env, 100, 1)
    mc_glie.train(50000)
    plot_value_func(mc_glie.get_values())