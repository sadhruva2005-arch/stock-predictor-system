# Stock Portfolio Predictor System

A **Reinforcement Learning** portfolio optimisation system that trains a PPO agent to allocate weights across multiple assets, balancing returns against risk and transaction costs — all inside a custom Gymnasium environment.

---

## Overview

Traditional portfolio theory (Markowitz, CAPM) assumes static distributions. This project frames portfolio allocation as a **sequential decision-making problem**, where an RL agent learns a dynamic allocation policy directly from simulated return streams.

The agent is rewarded for high risk-adjusted returns and penalised for both volatility and excessive rebalancing (transaction costs).

---

## Project Structure

```
stock-predictor-system/
│
├── main.py                  # Entry point — verifies all dependencies
│
├── decision/
│   ├── __init__.py          # PortfolioEnv (full version with transaction costs)
│   ├── env.py               # PortfolioEnv (base version)
│   ├── agent.py             # PPO agent setup via Stable Baselines3
│   └── train.py             # Training loop — runs PPO for 20,000 timesteps
│
└── Documents/
    └── WGAN/                # Reference generative model experiments
```

---

## Environment: `PortfolioEnv`

A custom `gymnasium.Env` that simulates a multi-asset portfolio allocation task.

| Parameter | Default | Description |
|---|---|---|
| `n_assets` | 3 | Number of assets in the portfolio |
| `steps` | 200 | Episode length (trading steps) |
| `lambda_risk` | 5 | Risk aversion coefficient |
| `eta_cost` | 0.1 | Transaction cost penalty weight |

### Observation Space
Vector of shape `(n_assets,)` — the current-step asset returns, drawn from `N(0.001, 0.02)`.

### Action Space
Vector of shape `(n_assets,)` in `[0, 1]` — portfolio weights (automatically normalised to sum to 1).

### Reward Function

```
reward = portfolio_return
       - λ_risk  × portfolio_return²      ← variance penalty
       - η_cost  × Σ|wₜ - wₜ₋₁|          ← transaction cost penalty
```

The agent learns to maximise cumulative risk-adjusted return while minimising unnecessary rebalancing.

---

## Agent: PPO (Proximal Policy Optimization)

Built with [Stable Baselines3](https://stable-baselines3.readthedocs.io/).

| Hyperparameter | Value |
|---|---|
| Policy | `MlpPolicy` |
| Learning rate | `3e-4` |
| Steps per update | `2048` |
| Batch size | `64` |
| Total timesteps | `20,000` |

---

## Getting Started

### Install dependencies

```bash
pip install torch stable-baselines3 cvxpy gymnasium numpy
```

### Verify setup

```bash
python main.py
# All major libraries imported successfully 🚀
```

### Train the agent

```bash
python decision/train.py
```

This trains the PPO agent for 20,000 timesteps and saves the model as `ppo_portfolio_model.zip`.

---

## How It Works

```
Simulated Asset Returns (n_assets × steps)
              │
              ▼
     PortfolioEnv (Gymnasium)
              │
     Observation: current returns
              │
              ▼
        PPO Agent (MLP Policy)
              │
     Action: portfolio weights [w₁, w₂, w₃]
              │
              ▼
   Reward = return - risk - transaction cost
              │
              ▼
     Agent updates policy via PPO
              │
              ▼
   Saved model: ppo_portfolio_model.zip
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `torch` | Neural network backend |
| `stable-baselines3` | PPO RL algorithm |
| `gymnasium` | Custom environment interface |
| `cvxpy` | Convex optimisation (portfolio constraints) |
| `numpy` | Numerical computation |

---

## Author

**sadhruva2005-arch**
