import matplotlib as matplotlib
import matplotlib
import numpy as np
import sys
from blackjack import BlackjackEnv
from collections import defaultdict

if "../" not in sys.path:
  sys.path.append("../")
import plotting

matplotlib.style.use('ggplot')


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final value function
    V = defaultdict(float)

    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all states the we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        states_in_episode = set([tuple(x[0]) for x in episode])
        for state in states_in_episode:

            # Find the first occurance of the state in the episode
            first_occurence_idx = next(i for i, x in enumerate(episode) if x[0] == state)
            # Sum up all rewards since the first occurance
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]
        print(V)
    return V


def sample_policy(observation):
    """
    A policy that sticks if the player score is >= 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

env = BlackjackEnv()
V_10k = mc_prediction(sample_policy, env, num_episodes=100)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500)
plotting.plot_value_function(V_500k, title="500,000 Steps")

# a = Dataset.from_tensor_slices([11, 10, 13, 12, 13,  0,  0, 10, 12,  0, 10,  0,  0,  0,  0,  0,  0,  0,
#         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  5])
#
# a.flat_map(lambda x: Dataset.from_tensor_slices(x +1))
#
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#
#
# model = Sequential()
# # Input - Layer
# model.add(layers.Dense(10, activation = "relu", input_shape=(7, )))
# # Hidden - Layers
# model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
# model.add(layers.Dense(2, activation = "relu"))
# model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
# model.add(layers.Dense(1, activation = "sigmoid"))
# # Output- Layer
# model.summary()
#
# model.compile(
#  optimizer = "adam",
#  loss = "binary_crossentropy",
#  metrics = ["accuracy"]
# )
#
# from sklearn.model_selection import validation_curve, train_test_split
#
# # fit model
# history = model.fit(X_train, y_train, epochs=10, batch_size=150, validation_data=(X_test, y_test))
# # evaluate the model
#
# v_loss = history.history['val_loss']
# v_acc = history.history['val_accuracy']
# acc = history.history['accuracy']
# runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#
# # plt.plot(runs, v_loss)
# plt.plot(runs, v_acc)
# plt.plot(runs, acc)