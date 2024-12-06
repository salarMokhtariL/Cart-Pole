import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from IPython import display

env = gym.make('CartPole-v0', render_mode='rgb_array').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("CUDA Available: ", torch.cuda.is_available())
    print("Device Count: ", torch.cuda.device_count())
    print("Current Device: ", torch.cuda.current_device())
    print("Device Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

plt.ion()

Experience = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ExperienceBuffer(object):
    def __init__(self, max_capacity):
        self.buffer = deque([], maxlen=max_capacity)

    def add_experience(self, *experience_details):
        """Save an experience tuple"""
        self.buffer.append(Experience(*experience_details))

    def get_random_samples(self, sample_size):
        return random.sample(self.buffer, sample_size)

    def __len__(self):
        return len(self.buffer)
    
class Deep_Q_Network(nn.Module):
    def __init__(self, frame_height, frame_width, action_count):
        super(Deep_Q_Network, self).__init__()
        self.conv_layer1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)  # If gray Scale 3 to 1
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv_layer2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv_layer3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)

        def compute_output_size(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        processed_width = compute_output_size(compute_output_size(compute_output_size(frame_width)))
        processed_height = compute_output_size(compute_output_size(compute_output_size(frame_height)))
        flattened_input_size = processed_width * processed_height * 32
        self.final_layer = nn.Linear(flattened_input_size, action_count)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.batch_norm1(self.conv_layer1(x)))
        x = F.relu(self.batch_norm2(self.conv_layer2(x)))
        x = F.relu(self.batch_norm3(self.conv_layer3(x)))
        return self.final_layer(x.view(x.size(0), -1))

    
resize = T.Compose([
    T.ToPILImage(),
    T.Grayscale(),  # Convert to grayscale
    T.Resize(40, interpolation=Image.BICUBIC),  # BICUBIC is a higher-quality resizing technique (uses cubic polynomials to interpolate pixel values during resizing)
    T.ToTensor()
])


def calculate_cart_screen_position(screen_width):
    total_world_width = env.x_threshold * 2
    screen_to_world_scale = screen_width / total_world_width
    cart_position_in_world = env.state[0]
    cart_screen_position = int(cart_position_in_world * screen_to_world_scale + screen_width / 2.0)
    return cart_screen_position  # Centered screen position of the cart

def capture_environment_screen():
    raw_screen = env.render().transpose((2, 0, 1))  # Capture raw screen
    _, screen_height, screen_width = raw_screen.shape
    processed_screen = raw_screen[:, int(screen_height * 0.38):int(screen_height * 0.9)]
    focus_area_width = int(screen_width * 0.5)
    cart_screen_position = calculate_cart_screen_position(screen_width)

    # Adjust slice to center cart on screen
    if cart_screen_position < focus_area_width // 2:
        screen_slice = slice(focus_area_width)
    elif cart_screen_position > (screen_width - focus_area_width // 2):
        screen_slice = slice(-focus_area_width, None)
    else:
        screen_slice = slice(cart_screen_position - focus_area_width // 2,
                             cart_screen_position + focus_area_width // 2)
    
    centered_screen = processed_screen[:, :, screen_slice]
    centered_screen = np.ascontiguousarray(centered_screen, dtype=np.float32) / 255
    centered_screen_tensor = torch.from_numpy(centered_screen)

    # Resize and add a batch dimension for grayscale
    return resize(centered_screen_tensor).unsqueeze(0)


env.reset()
plt.figure()
plt.imshow(capture_environment_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
plt.title('One Sample of captured frame')
plt.show()

epsilon = 0.01 # epsilon greedy exploration-exploitation (higher more random)
EPS_START = 1 
EPS_END = 0.001
TARGET_UPDATE = 20  # Increased target network update frequency
BATCH_SIZE = 128  # Increased batch size for more stable training
GAMMA = 0.99  # Slightly increased discount factor for long-term reward focus

init_screen = capture_environment_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = Deep_Q_Network(screen_height, screen_width, n_actions).to(device)
target_net = Deep_Q_Network(screen_height, screen_width, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)  # Changed to Adam optimizer for better convergence
replay_buffer = ExperienceBuffer(50000)  # Increased buffer size for better experience replay

steps_done = 0
epsilon_values = [] 

def select_action(state, i_episode):
    global steps_done
    sample = random.random()
    epsilon_threshold = EPS_START * np.exp(-epsilon_decay_rate * i_episode)
    epsilon_values.append(epsilon_threshold)

    steps_done += 1
    if sample > epsilon_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    

episode_durations = []

def plot_training_durations():
    # Create a new figure for the plot
    fig = plt.figure(2)
    plt.clf()

    # Convert episode rewards to tensors
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)

    # Plot rewards per episode
    plt.subplot(3, 1, 1)
    plt.title('Training Progress (Per Episode)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(rewards_t.numpy(), label="Total Reward")
    
    # Add smoothed rewards
    if len(episode_rewards) >= 50:  # Ensure enough data for smoothing
        smoothed_rewards = moving_average(episode_rewards, window_size=50)
        plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label="Smoothed Rewards", color='orange')
    plt.legend()

    # Plot losses per step
    plt.subplot(3, 1, 2)
    plt.title('Loss (Per Step)')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.plot(losses, label="Loss", color='red')
    plt.legend()

    # Plot epsilon decay per step
    plt.subplot(3, 1, 3)
    plt.title('Epsilon Decay (Per Step)')
    plt.xlabel('Training Step')
    plt.ylabel('Epsilon')
    plt.plot(epsilon_values, label="Epsilon Decay", color='blue')
    plt.legend()

    # Adjust layout to increase space between plots
    plt.subplots_adjust(hspace=0.5)  # Adjust spacing between rows

    # Pause briefly to update the plot
    plt.pause(0.01)

    # Clear the current output and display the updated plot
    display.display(plt.gcf())

def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def optimize_policy_net():
    # Check if enough transitions are available in replay_buffer
    if len(replay_buffer) < BATCH_SIZE:
        return

    # Sample a batch of transitions
    transitions = replay_buffer.get_random_samples(BATCH_SIZE)
    batch = Experience(*zip(*transitions))

    # Create a mask for non-final next states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    # Concatenate batch tensors
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q values for the current state-action pairs
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute Q values for the next states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the policy network
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    # Track the loss for plotting
    losses.append(loss.item())

# Training loop
num_episodes = 1500

episode_rewards = []  # Store total rewards per episode
losses = []

epsilon_decay_rate = -np.log(EPS_END / EPS_START) / num_episodes

for i_episode in range(num_episodes):
    env.reset()
    last_screen = capture_environment_screen()
    current_screen = capture_environment_screen()
    state = current_screen - last_screen

    total_reward = 0  # Track total reward for this episode

    for t in count():
        # Select and perform an action
        action = select_action(state, i_episode)
        _, reward, done, _, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        total_reward += reward.item()  # Accumulate rewards
     
        # Observe new state
        last_screen = current_screen
        current_screen = capture_environment_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store transition in replay buffer
        replay_buffer.add_experience(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Optimize the policy network
        optimize_policy_net()
        if done:
            episode_durations.append(t + 1)
            episode_rewards.append(total_reward)  # Save total reward for this episode
            if i_episode % 10 == 0:
                plot_training_durations()

            break

        # Update target network
        if t % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
    print(i_episode,'/',num_episodes, ', R= ', episode_rewards[i_episode])

def run_demo(policy_net, num_episodes=1):
    """
    Run a test demo using the trained policy network.

    Args:
    - policy_net: The trained DQN model.
    - num_episodes: Number of test episodes to run.
    """
    # Create a new environment with 'human' render mode for visualization
    demo_env = gym.make('CartPole-v0', render_mode='human').unwrapped

    for episode in range(num_episodes):
        # Reset environment and initialize the state
        demo_env.reset()
        last_screen = capture_environment_screen()
        current_screen = capture_environment_screen()
        state = current_screen - last_screen
        total_reward = 0  # Track total reward for the episode

        print(f"Demo Episode {episode + 1} started...")

        for t in count():
            # Render the environment for visualization
            demo_env.render()

            # Select the best action based on the policy network (no exploration)
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)  # Greedy action selection

            # Perform the action in the environment
            _, reward, done, _, _ = demo_env.step(action.item())
            reward = torch.tensor([reward], device=device)
            total_reward += reward.item()

            # Observe the next state
            last_screen = current_screen
            current_screen = capture_environment_screen()
            if not done:
                next_state = current_screen - last_screen
                state = next_state  # Update state for the next step
            else:
                print(f"Demo Episode {episode + 1}: Total Reward = {total_reward}")
                break

    # Close the demo environment
    demo_env.close()

print('Training is finished')
run_demo(policy_net, num_episodes=10)
env.render()
env.close()
plt.ioff()
plt.show()