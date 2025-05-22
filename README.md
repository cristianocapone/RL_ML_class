🐜 Ant-v4 Reinforcement Learning with Custom Recurrent Network

This repository contains a reinforcement learning (RL) agent that learns to walk in the MuJoCo Ant-v4 environment using a custom recurrent neural network architecture with biologically-inspired dynamics.

https://gym.openai.com/envs/Ant-v4/

🚀 Features

✅ Uses Ant-v4 MuJoCo environment from OpenAI Gym
✅ Custom RL architecture with:
Leaky integrator dynamics
Reinforcement-driven Hebbian plasticity
Actor-Critic-like updates (with value prediction)
✅ Periodic rendering and video saving
✅ Automatic training progress plots
✅ Save/load model with pickle
✅ Highly customizable network and training parameters
🧠 Architecture

The network (net) is a custom time-dependent recurrent model where:

Neurons have leaky membrane potentials
Synaptic traces and output plasticity evolve over time
Policy is learned through continuous interaction with the environment and internal credit assignment
🏁 Getting Started

🔧 Requirements
Install MuJoCo and Gym:

pip install gym[mujoco] matplotlib numpy tqdm
MuJoCo setup (if not already done):

pip install mujoco
📂 Run Training
# Inside your notebook or script
!git clone https://github.com/yourusername/ant-v4-recurrent-rl.git
%cd ant-v4-recurrent-rl
!python train.py
(Make sure train.py contains the code you posted, or split it into modular files)

📊 Training Outputs

rewards_ant_hier_transfer.png — periodic training plots
AntMujocoEnv-v4*.mp4 — rendered episodes saved as videos
transfer_new.pickle — trained model weights
trajectory_0.npy — reward trajectory data
📝 Parameters

You can tune parameters like:

episode_duration: episode length
act_variance: action exploration noise
alpha_pg, alpha_rout: learning rates
out_dim: output action dimension (default = 8 for Ant)
N, I, O, T: network size
📦 File Structure

ant-v4-recurrent-rl/
├── train.py                # Main training loop
├── net.py                  # Custom network definition (assumed)
├── transfer_new.pickle     # Trained model (saved periodically)
├── trajectory_0.npy        # Reward trajectory
├── rewards_ant_hier_transfer.png
└── AntMujocoEnv-v4*.mp4    # Rendered video episodes

💾 Model Persistence

To resume training or transfer learning:

with open('transfer.pickle', 'rb') as f:
    network = pickle.load(f)
📽️ Example Rendering

Set render_every = 100 to enable periodic rendering and save an .mp4 of the ant agent walking.

🤖 Future Plans

Add Gym wrapper to support other MuJoCo environments
Modularize network into reusable class
Support PyTorch version
Upload pretrained weights
📜 License

MIT License
