import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import numpy as np
import math
import reinforcement_learning as rl

env_name = 'Breakout-v0'
rl.update_paths(env_name=env_name)

# Create Agent
# The Agent-class implements the main loop for playing the game,
# recording data and optimizing the Neural Network.
# We create an object-instance and need to set training=True
# because we want to use the replay-memory to record states
# and Q-values for plotting further below.

agent = rl.Agent(env_name=env_name,
                 training=True,
                 render=True,
                 use_logging=False)

model = agent.model
replay_memory = agent.replay_memory

# Training
# The agent's run() function is used to play the game.
# This uses the Neural Network to estimate Q-values and
# hence determine the agent's actions. If training==True
# then it will also gather states and Q-values in the replay-memory
# and train the Neural Network when the replay-memory is sufficiently full.
# You can set num_episodes=None if you want an infinite loop
# that you would stop manually with ctrl-c.
# In this case we just set num_episodes=1
# because we are not actually interested in training the Neural Network any further,
# we merely want to collect some states and Q-values in the replay-memory
# so we can plot them below.

agent.run(num_episodes=1)

# Training Progress: Reward

# Data is being logged during training so we can plot the progress afterwards.
# The reward for each episode and a running mean of the last 30 episodes are logged to file.
# Basic statistics for the Q-values in the replay-memory are also logged to file before each optimization run.

log_q_values = rl.LogQValues()
log_reward = rl.LogReward()

log_q_values.read()
log_reward.read()

plt.plot(log_reward.count_states, log_reward.episode, label='Episode Reward')
plt.plot(log_reward.count_states, log_reward.mean, label='Mean of 30 episodes')
plt.xlabel('State-Count for Game Environment')
plt.legend()
plt.show()

# Training Progress: Q-Values
plt.plot(log_q_values.count_states, log_q_values.mean, label='Q-Value Mean')
plt.xlabel('State-Count for Game Environment')
plt.legend()
plt.show()

# Testing
# When the agent and Neural Network is being trained,
# the so-called epsilon-probability is typically decreased from 1.0 to 0.1
# over a large number of steps, after which the probability is held fixed at 0.1.
# This means the probability is 0.1 or 10% that the
# agent will select a random action in each step,
# otherwise it will select the action that has the highest Q-value.
# This is known as the epsilon-greedy policy.
# The choice of 0.1 for the epsilon-probability is a compromise
# between taking the actions that are already known to be good,
# versus exploring new actions that might lead to even higher rewards
# or might lead to death of the agent.

print(agent.epsilon_greedy.epsilon_testing)

agent.training = True
agent.reset_episode_rewards()
agent.render = True
agent.run(num_episodes=1)

agent.reset_episode_rewards()
agent.render = False
agent.run(num_episodes=30)

rewards = agent.episode_rewards
print("Rewards for {0} episodes:".format(len(rewards)))
print("- Min:   ", np.min(rewards))
print("- Mean:  ", np.mean(rewards))
print("- Max:   ", np.max(rewards))
print("- Stdev: ", np.std(rewards))

# plot a histogram
_ = plt.hist(rewards, bins=30)

# Plot examples of states form the game-environment and the Q-values that estimated by the NN.

# Prints the Q-values for a given index in the replay-memory
def print_q_values(idx):
    # Get the Q-values and action from the replay-memory
    q_values = replay_memory.q_values[idx]
    action = replay_memory.actions[idx]

    print("Action:      Q-Value:")
    print("=====================")

    # Print all the actions and their Q-values.
    for i, q_value in enumerate(q_values):
        # Used to display which action was taken.
        if i == action:
            action_taken = "(Action Taken)"
        else:
            action_taken = ""

        # Text-name of the action.
        action_name = agent.get_action_name(i)

        print("{0:12}{1:.3f} {2}".format(action_name, q_value,
                                         action_taken))

    # Newline.
    print()

# Plots a state from the replay-memory and optionally prints the Q-values
def plot_state(idx, print_q = True):
    # Get the state from the replay-memory
    state = replay_memory.states[idx]

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(1, 2)

    # Plot the image from the game-environment.
    ax = axes.flat[0]
    ax.imshow(state[:, :, 0], vmin=0, vmax=255,
              interpolation='lanczos', cmap='gray')

    # Plot the motion-trace.
    ax = axes.flat[1]
    ax.imshow(state[:, :, 1], vmin=0, vmax=255,
              interpolation='lanczos', cmap='gray')

    # This is necessary if we show more than one plot in a single Notebook cell.
    plt.show()

    # Print the Q-values.
    if print_q:
        print_q_values(idx=idx)

num_used = replay_memory.num_used
q_values = replay_memory.q_values[0:num_used, :]
q_values_min = q_values.min(axis=1)
q_values_max = q_values.max(axis=1)
q_values_dif = q_values_max - q_values_min

# Highest Reward
idx = np.argmax(replay_memory.rewards)
for i in range(-5, 3):
    plot_state(idx=idx+i)

# Highest Q-Value
# This means that the agent has high expectation that
# several points will be scored in the following steps.
# Note that the Q-values decrease significantly after the points have been scored.

idx = np.argmax(q_values_max)
for i in range(0, 5):
    plot_state(idx=idx+i)

# Loss of life
idx = np.argmax(replay_memory.end_life)
for i in range(-10, 0):
    plot_state(idx=idx+i)

# Greatest Difference in Q-Values
# This example shows the state where there is the greatest difference in Q-values,
# which means that the agent believes one action will be much more beneficial than another.
# But because the agent uses the Epsilon-greedy policy,
# it sometimes selects a random action instead.
idx = np.argmax(q_values_dif)
for i in range(0, 5):
    plot_state(idx=idx+i)

# Smallest Difference in Q-Values
# This example shows the state where there is the smallest difference in Q-values,
# which means that the agent believes it does not really matter which action it selects,
# as they all have roughly the same expectations for future rewards.
# The Neural Network estimates these Q-values and they are not precise.
# The differences in Q-values may be so small that they fall within the error-range of the estimates.
idx = np.argmin(q_values_dif)
for i in range(0, 5):
    plot_state(idx=idx+i)

# Output of Convolutional Layers
def plot_layer_output(model, layer_name, state_index, inverse_cmap=False):
    """
    Plot the output of a convolutional layer.

    :param model: An instance of the NeuralNetwork-class.
    :param layer_name: Name of the convolutional layer.
    :param state_index: Index into the replay-memory for a state that
                        will be input to the Neural Network.
    :param inverse_cmap: Boolean whether to inverse the color-map.
    """

    # Get the given state-array from the replay-memory.
    state = replay_memory.states[state_index]

    # Get the output tensor for the given layer inside the TensorFlow graph.
    # This is not the value-contents but merely a reference to the tensor.
    layer_tensor = model.get_layer_tensor(layer_name=layer_name)

    # Get the actual value of the tensor by feeding the state-data
    # to the TensorFlow graph and calculating the value of the tensor.
    values = model.get_tensor_value(tensor=layer_tensor, state=state)

    # Number of image channels output by the convolutional layer.
    num_images = values.shape[3]

    # Number of grid-cells to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_images))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids, figsize=(10, 10))

    print("Dim. of each image:", values.shape)

    if inverse_cmap:
        cmap = 'gray_r'
    else:
        cmap = 'gray'

    # Plot the outputs of all the channels in the conv-layer.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid image-channels.
        if i < num_images:
            # Get the image for the i'th output channel.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap=cmap)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

idx = np.argmax(q_values_max)
plot_state(idx=idx, print_q=False)

plot_layer_output(model=model, layer_name='layer_conv1', state_index=idx, inverse_cmap=False)
plot_layer_output(model=model, layer_name='layer_conv2', state_index=idx, inverse_cmap=False)
plot_layer_output(model=model, layer_name='layer_conv3', state_index=idx, inverse_cmap=False)


def plot_conv_weights(model, layer_name, input_channel=0):
    """
    Plot the weights for a convolutional layer.

    :param model: An instance of the NeuralNetwork-class.
    :param layer_name: Name of the convolutional layer.
    :param input_channel: Plot the weights for this input-channel.
    """

    # Get the variable for the weights of the given layer.
    # This is a reference to the variable inside TensorFlow,
    # not its actual value.
    weights_variable = model.get_weights_variable(layer_name=layer_name)

    # Retrieve the values of the weight-variable from TensorFlow.
    # The format of this 4-dim tensor is determined by the
    # TensorFlow API. See Tutorial #02 for more details.
    w = model.get_variable_value(variable=weights_variable)

    # Get the weights for the given input-channel.
    w_channel = w[:, :, input_channel, :]

    # Number of output-channels for the conv. layer.
    num_output_channels = w_channel.shape[2]

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w_channel)
    w_max = np.max(w_channel)

    # This is used to center the colour intensity at zero.
    abs_max = max(abs(w_min), abs(w_max))

    # Print statistics for the weights.
    print("Min:  {0:.5f}, Max:   {1:.5f}".format(w_min, w_max))
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w_channel.mean(),
                                                 w_channel.std()))

    # Number of grids to plot.
    # Rounded-up, square-root of the number of output-channels.
    num_grids = math.ceil(math.sqrt(num_output_channels))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_output_channels:
            # Get the weights for the i'th filter of this input-channel.
            img = w_channel[:, :, i]

            # Plot image.
            ax.imshow(img, vmin=-abs_max, vmax=abs_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

plot_conv_weights(model=model, layer_name='layer_conv1', input_channel=0)
plot_conv_weights(model=model, layer_name='layer_conv1', input_channel=1)
plot_conv_weights(model=model, layer_name='layer_conv2', input_channel=0)
plot_conv_weights(model=model, layer_name='layer_conv3', input_channel=0)