import numpy as np
import torch as torch
import torch.nn.functional as F
import gym
from torch.utils.tensorboard import SummaryWriter
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Actor Network
class ActorNetwork(torch.nn.Module):
    def __init__(self, alpha, state_dim, fc1_dim, fc2_dim, n_actions):
        super(ActorNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, fc1_dim)
        self.fc2 = torch.nn.Linear(fc1_dim, fc2_dim)
        self.mu = torch.nn.Linear(fc2_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu(x))
        return mu

# Define the Critic Network
class CriticNetwork(torch.nn.Module):
    def __init__(self, beta, state_dim, fc1_dim, fc2_dim, n_actions):
        super(CriticNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + n_actions, fc1_dim)
        self.fc2 = torch.nn.Linear(fc1_dim, fc2_dim)
        self.q = torch.nn.Linear(fc2_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

# Define the DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, n_actions, alpha=0.0001, beta=0.0001, gamma=0.99, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, state_dim, 400, 300, n_actions)
        self.critic = CriticNetwork(beta, state_dim, 400, 300, n_actions)
        self.target_actor = ActorNetwork(alpha, state_dim, 400, 300, n_actions)
        self.target_critic = CriticNetwork(beta, state_dim, 400, 300, n_actions)

        self.update_network_parameters(tau=tau)

    def choose_action(self, observation):
        if isinstance(observation, tuple):
            observation = observation[0]
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        actions = self.actor.forward(state)
        noise = torch.rand(self.n_actions).to(self.actor.device) * 0.1  #FIXME: Add exploration noise with decay
        action = actions + noise
        action = action.detach().cpu().numpy()[0]  # Convert to NumPy array
        
        # clipping for LunarLanderContinuous-v2
        action[0] = np.clip(action[0], 0, 1)  # Clip main engine to [0, 1]
        action[1] = np.clip(action[1], -1, 1)  # Clip side engine to [-1, 1]
        return action
 
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_models(self):
        torch.save(self.actor.state_dict(), 'actor.pth')
        torch.save(self.critic.state_dict(), 'critic.pth')
        torch.save(self.target_actor.state_dict(), 'target_actor.pth')
        torch.save(self.target_critic.state_dict(), 'target_critic.pth')

    def load_models(self):
        self.actor.load_state_dict(torch.load('actor.pth'))
        self.critic.load_state_dict(torch.load('critic.pth'))
        self.target_actor.load_state_dict(torch.load('target_actor.pth'))
        self.target_critic.load_state_dict(torch.load('target_critic.pth'))

    def learn(self, memory, batch_size):
        if memory.mem_cntr < batch_size:
            return

        states, actions, rewards, states_next, dones = memory.sample_buffer(batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        states_next = torch.tensor(states_next, dtype=torch.float).to(self.actor.device)
        dones = torch.tensor(dones).to(self.actor.device)

        # Critic loss
        target_actions = self.target_actor.forward(states_next)
        critic_value_next = self.target_critic.forward(states_next, target_actions).flatten()
        critic_value_next[dones] = 0.0
        target = rewards + self.gamma * critic_value_next
        critic_value = self.critic.forward(states, actions).flatten()
        critic_loss = F.mse_loss(target, critic_value)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Actor loss
        actor_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        self.update_network_parameters()
        
    

# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size, state_dim, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.new_state_memory = np.zeros((self.mem_size, state_dim))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state[0]
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_next = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_next, dones

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


# Main Training Loop
if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = DDPGAgent(state_dim, n_actions)
    memory = ReplayBuffer(1000000, state_dim, n_actions)

    PRINT_INTERVAL = 1
    N_episodes = 1000
    MAX_STEPS = 1000
    batch_size = 64
    total_steps = 0
    score_history = []
    writer = SummaryWriter('DDPG_logs')
    dummy_input = torch.zeros((1, 8)).to(device)
    writer.add_graph(agent.actor, dummy_input)
    

    for i in range(N_episodes):
        state = env.reset()
        score = 0
        done = False
        episode_step = 0
        while not done:
            if episode_step >= MAX_STEPS:
                break
            action = agent.choose_action(state)
            state_next, reward, done, info, tmp = env.step(action)  # Ensure action is clipped and in correct format
            memory.store_transition(state, action, reward, state_next, done)
            agent.learn(memory, batch_size)

            state = state_next
            score += reward
            total_steps += 1
            episode_step += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
            agent.save_models()
            writer.add_scalars("episode rewars/", {"training_reward": avg_score}, total_steps)
            # writer.add_scalars("epsilon/", {"epsilon": agent.epsilon}, total_env_steps)
    
    writer.close()
    
    # Export TensorBoard data to PNG files
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "png_exports", "run001")
    print(f"Exporting TensorBoard metrics to PNG files at {output_dir}")
    export_tensorboard_to_png(str('DDPG_logs'), output_dir)

