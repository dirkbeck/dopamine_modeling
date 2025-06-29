import numpy as np
import random
from tqdm import tqdm
import warnings
from environment import Environment
import agents as agnt
import plotting_functions as plt_funs

warnings.filterwarnings('ignore')
random.seed(0)
np.random.seed(0)

num_sims = 10
episodes = 100


def run_single_simulation(sim_id, episodes=1000, use_direct_env_input=False):
    env = Environment()
    agents = [
        agnt.ActorCriticGovernorAgent(env, use_direct_env_input=use_direct_env_input),
        agnt.ActorCriticAgent(env, use_direct_env_input=use_direct_env_input),
        agnt.ActorGovernorAgent(env, use_direct_env_input=use_direct_env_input),
        agnt.RandomAgent(env)
    ]
    rewards_all = {}
    agent_metrics = {}

    print(f"sim {sim_id + 1}/{num_sims}")

    for agent in agents:
        rewards = agent.learn(episodes)
        rewards_all[agent.name] = rewards

        if hasattr(agent, 'get_layer_impacts'):
            if agent.name not in agent_metrics: agent_metrics[agent.name] = {}
            agent_metrics[agent.name].update(agent.get_layer_impacts())

        if hasattr(agent, 'get_region_component_data'):
            if agent.name not in agent_metrics: agent_metrics[agent.name] = {}
            agent_metrics[agent.name]['region_component_usage'] = agent.get_region_component_data()

    metrics = {}
    for agent_name, rewards in rewards_all.items():
        num_rewards_for_avg = min(100, len(rewards))
        if num_rewards_for_avg > 0:
            final_performance = np.mean(rewards[-num_rewards_for_avg:])
        else:
            final_performance = 0
        metrics[agent_name] = final_performance

    return metrics, rewards_all, agent_metrics


def run_multiple_simulations(num_simulations=10, episodes=1000, use_direct_env_input=False):

    results = {
        'ACG': [], 'AC': [], 'AG': [], 'Random': []
    }
    all_rewards = {
        'ACG': [], 'AC': [], 'AG': [], 'Random': []
    }
    all_agent_metrics = []

    for i in tqdm(range(num_simulations)):
        metrics, rewards, agent_metrics_single_run = run_single_simulation(
            i, episodes, use_direct_env_input=use_direct_env_input
        )

        for agent_name, metric in metrics.items():
            if agent_name in results:
                results[agent_name].append(metric)
            if agent_name in all_rewards:
                all_rewards[agent_name].append(rewards[agent_name])

        all_agent_metrics.append(agent_metrics_single_run)

    return results, all_rewards, all_agent_metrics


results, all_rewards, all_agent_metrics = run_multiple_simulations(num_simulations=num_sims,
                                                                   episodes=episodes,
                                                                   use_direct_env_input=False)

direct_results, direct_rewards, direct_metrics = run_multiple_simulations(
    num_simulations=num_sims, episodes=episodes, use_direct_env_input=True
)

plt_funs.plot_boxplots(results)
plt_funs.plot_layer_component_relationships(all_agent_metrics)
plt_funs.plot_performance_comparison(all_rewards, direct_rewards, episodes=episodes)