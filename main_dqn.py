import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tensorflow.python.summary.summary_iterator import summary_iterator
from pathlib import Path
import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EpsilonDecayScheduler():
    # linearly decays the epsilon from start to finish over the aneal time.
    def __init__(self, eps_start, eps_finish, eps_aneal_time):
        self.eps_start = eps_start
        self.eps_finish = eps_finish
        self.eps_aneal_time = eps_aneal_time
        self.dd = (self.eps_start - self.eps_finish) / self.eps_aneal_time

    def get_epsilon(self, current_time_step):
        return max(self.eps_finish, self.eps_start - self.dd * current_time_step)


class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.to(self.device)

    def forward(self, state):
        # Cast the input state tensor to the same data type as the weight tensor of fc1
        state = state.to(self.fc1.weight.dtype)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


class Agent:
    def __init__(self, gamma, epsilon_start, lr, input_dims, batch_size, n_actions, epsilon_aneal_time,
                 replay_buffer_size, epsilon_end):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_aneal_time = epsilon_aneal_time
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.replay_buffer_couter = 0
        self.iter_cntr = 0

        # Evaluation network (Q_eval)
        self.Q_eval = DeepQNetwork(input_dims=input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions).to(device)
        # Target network (Q_target) - created using deepcopy of Q_eval
        self.Q_target = deepcopy(self.Q_eval).to(device)  # Use deepcopy to create Q_target

        # Optimizer and loss function moved to the Agent class
        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.total_num_network_params = sum(p.numel() for p in self.Q_eval.parameters())
        print(f"##### total number of network params = {self.total_num_network_params}")

        self.epsilon_decay_scheduler = EpsilonDecayScheduler(self.epsilon_start, self.epsilon_end, self.epsilon_aneal_time)
        self.state_memory = np.zeros((self.replay_buffer_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.replay_buffer_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.replay_buffer_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.replay_buffer_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.replay_buffer_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.replay_buffer_couter % self.replay_buffer_size
        self.state_memory[index] = state[0]
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal
        self.replay_buffer_couter += 1

    def choose_action(self, observation, curr_env_step=None, explore=False):
        if explore:
            self.epsilon = self.epsilon_decay_scheduler.get_epsilon(curr_env_step)
            if np.random.random() > self.epsilon:
                if isinstance(observation, tuple):
                    observation = observation[0]
                state = torch.tensor([observation]).to(self.Q_eval.device)
                actions = self.Q_eval.forward(state)
                action = torch.argmax(actions).item()
            else:
                action = np.random.choice(self.action_space)
        else:
            state = torch.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        return action

    def learn(self):
        if self.replay_buffer_couter < self.batch_size:
            return

        self.optimizer.zero_grad()

        max_mem = min(self.replay_buffer_couter, self.replay_buffer_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        # Compute Q-values for current states using the evaluation network
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]

        # Compute Q-values for next states using the target network
        with torch.no_grad():  # No need to compute gradients for the target network
            q_next = self.Q_target.forward(new_state_batch)
            q_next[terminal_batch] = 0.0

        # Compute target Q-values
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        # Compute loss and update the evaluation network
        loss = self.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.optimizer.step()


def export_tensorboard_to_png(log_dir, output_dir):
    """
    Export TensorBoard metrics to PNG files
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_dir: Directory to save PNG files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))
    
    # Dictionary to store metrics
    metrics = {}
    
    # Parse each event file
    for event_file in event_files:
        for event in summary_iterator(event_file):
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    if value.tag not in metrics:
                        metrics[value.tag] = {"steps": [], "values": []}
                    metrics[value.tag]["steps"].append(event.step)
                    metrics[value.tag]["values"].append(value.simple_value)
    
    # Generate a plot for each metric
    for tag, data in metrics.items():
        plt.figure(figsize=(10, 6))
        plt.plot(data["steps"], data["values"])
        plt.title(tag)
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.grid(True)
        
        # Clean tag name for file naming
        clean_tag = tag.replace('/', '_').replace(' ', '_')
        save_path = os.path.join(output_dir, f"{clean_tag}.png")
        
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    print(f"Exported {len(metrics)} metrics to {output_dir}")


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    # There are four discrete actions available: do nothing, fire left engine, fire main engine, fire right engin
    # For more info visit:
    # https://www.gymlibrary.dev/environments/box2d/lunar_lander/

    #env = gym.make('LunarLanderContinuous-v2')  #  the continuous version

    n_episodes = 10000
    # n_episodes = 10
    target_update_interval_episode = 200

    agent = Agent(gamma=0.99, epsilon_start=1.0, lr=0.0005, input_dims=[8], batch_size=64, n_actions=4, epsilon_aneal_time = 4.5E5, 
                  epsilon_end=0.01, replay_buffer_size = int(2E5) )
    

    logs_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/logs")
    print(f"log directory = {logs_dir}")
    if not logs_dir.exists():
        curr_runn = "run001"
    else:
        existing_run_nums = [int(str(folder.name).split("run")[1]) for folder in logs_dir.iterdir() if str(folder.name).startswith("run")]
        if len(existing_run_nums) ==0:
            curr_runn = "run001"
        else:
            curr_runn= "run"+str(max(existing_run_nums)+1).zfill(3)
    logs_dir = logs_dir / curr_runn
    if not logs_dir.exists():
        os.makedirs(str(logs_dir))

    writer = SummaryWriter(logs_dir)
    dummy_input = torch.zeros((1, 8)).to(device)
    writer.add_graph(agent.Q_eval, dummy_input)

    total_env_steps = 0
    for i in range(n_episodes):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation, curr_env_step = total_env_steps, explore = True)
            observation_, reward, done, info, tmp = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            agent.learn()
            observation = observation_
            total_env_steps += 1
        if i % target_update_interval_episode ==0:
            agent.Q_target.load_state_dict(deepcopy(agent.Q_eval.state_dict()))  # Use deepcopy to make independent copy of the object and all its nested objects.
            print(f'target network updated.')
            
        writer.add_scalars("episode rewars/", {"training_reward": score}, total_env_steps)
        writer.add_scalars("epsilon/", {"epsilon": agent.epsilon}, total_env_steps)

        print(f"episode: {i}, env_step_num = {total_env_steps}, score = {score}")
    writer.close()
    
    # Export TensorBoard data to PNG files
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "png_exports", curr_runn)
    print(f"Exporting TensorBoard metrics to PNG files at {output_dir}")
    export_tensorboard_to_png(str(logs_dir), output_dir)
   

