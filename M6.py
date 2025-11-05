'''Reinforcement Learning (Any one)
A. Implement Reinforcement Learning using an example of a maze environment that the
agent needs to explore.'''



import numpy as np
import random
import matplotlib.pyplot as plt

# Define the environment (Maze)
class MazeEnvironment:
    def __init__(self):
        # 0: Empty cell, 1: Wall, 2: Start, 3: Goal
        self.maze = np.array([
            [2, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 1, 3]
        ])
        self.start_pos = (0, 0)  # Start position (matches the '2' in the maze)
        self.agent_pos = self.start_pos  # Agent starts here

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def get_state(self):
        return self.agent_pos

    def step(self, action):
        row, col = self.agent_pos

        # Define movement actions: [Up, Down, Left, Right]
        if action == 0:  # Up
            next_pos = (row - 1, col)
        elif action == 1:  # Down
            next_pos = (row + 1, col)
        elif action == 2:  # Left
            next_pos = (row, col - 1)
        elif action == 3:  # Right
            next_pos = (row, col + 1)
        else:
            next_pos = self.agent_pos

        # Check boundaries and walls
        r, c = next_pos
        if (r < 0 or r >= self.maze.shape[0] or
            c < 0 or c >= self.maze.shape[1] or
            self.maze[next_pos] == 1):
            # Invalid move -> stay in place
            next_pos = self.agent_pos

        self.agent_pos = next_pos

        # Check if the agent reached the goal
        if self.maze[self.agent_pos] == 3:
            return self.agent_pos, 10, True  # Goal reached
        else:
            return self.agent_pos, -1, False  # Step penalty


# Define the Q-Learning agent
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=1.0, exploration_decay=0.99, min_exploration=0.01):
        self.env = env
        self.q_table = np.zeros((env.maze.shape[0], env.maze.shape[1], 4))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration

    def choose_action(self, state):
        # Epsilon-greedy strategy
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)
        else:
            r, c = state
            return int(np.argmax(self.q_table[r, c]))

    def learn(self, state, action, reward, next_state):
        r, c = state
        nr, nc = next_state
        best_next = np.argmax(self.q_table[nr, nc])
        td_target = reward + self.discount_factor * self.q_table[nr, nc, best_next]
        td_error = td_target - self.q_table[r, c, action]
        self.q_table[r, c, action] += self.learning_rate * td_error

    def train(self, episodes, verbose=False):
        rewards_per_episode = []
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done and steps < 500:  # safeguard max steps
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                steps += 1

            # Decay exploration rate but don't go below minimum
            self.exploration_rate = max(self.min_exploration,
                                        self.exploration_rate * self.exploration_decay)

            rewards_per_episode.append(total_reward)
            if verbose:
                print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {steps}, Epsilon = {self.exploration_rate:.3f}")

        print("Training complete!")
        return rewards_per_episode


# Helper to show the learned greedy path from start
def show_greedy_path(env, agent, max_steps=50):
    state = env.reset()
    path = [state]
    done = False
    steps = 0
    while not done and steps < max_steps:
        r, c = state
        action = int(np.argmax(agent.q_table[r, c]))
        next_state, reward, done = env.step(action)
        path.append(next_state)
        state = next_state
        steps += 1
        if state == path[-2]:  # stuck (no change)
            break
    return path, done


# Main
if __name__ == "__main__":
    env = MazeEnvironment()
    agent = QLearningAgent(env, learning_rate=0.1, discount_factor=0.9,
                           exploration_rate=1.0, exploration_decay=0.99, min_exploration=0.01)

    # Train for 200 episodes (you can change)
    rewards = agent.train(episodes=200, verbose=True)

    # Plot total reward per episode
    plt.figure(figsize=(8,4))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.grid(True)
    plt.show()

    # Test learned policy: show greedy path
    path, reached = show_greedy_path(env, agent, max_steps=50)
    print("\nAgent's path (sequence of positions):")
    for step_pos in path:
        print(step_pos)
    print("Reached goal?" , reached)
