🦖 Dino RL Game using Deep Q-Network (DQN)

A Chrome Dino-inspired game where an AI agent learns to survive using Reinforcement Learning and a Deep Q-Network (DQN).

The project features:

🎮 Chrome-style GUI built with Pygame
🧠 Reinforcement Learning using PyTorch
📈 Dynamic difficulty scaling
🌵 Randomized obstacle generation
💾 Persistent model saving/loading
⚡ Real-time training and evaluation modes

The agent learns optimal jump timing through interaction with the environment using:

Experience Replay
Target Networks
Epsilon-Greedy Exploration

Over multiple episodes, the AI improves its survival strategy and continues learning from previously saved models.

🚀 Features
Reinforcement Learning-based gameplay
Persistent learning across runs
Dynamic obstacle spacing and increasing difficulty
Pixel-style animated dino
Dark-mode Chrome Dino aesthetic
Train mode and play-only mode
Automatic checkpoint saving
🛠 Technologies Used
Python
PyTorch
Pygame
NumPy
▶ Run the Project
Training Mode
python dino_rl_full.py
Play Mode (uses saved model)
python dino_rl_full.py --play
