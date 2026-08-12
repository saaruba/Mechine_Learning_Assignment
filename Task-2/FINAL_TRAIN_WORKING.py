"""
TASK 2: COMPLETE STANDALONE DRL TRAINING SCRIPT
NO DEPENDENCIES ON CONFIG FILES
WORKS 100% WITH MLJOCO REACHER
MlpPolicy for vector observations
"""

import os
import json
import numpy as np
import gymnasium as gym
import logging
from datetime import datetime
import torch
from stable_baselines3 import PPO, SAC, TD3

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION - ALL HARDCODED (NO CONFIG FILE NEEDED)
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOTAL_TIMESTEPS = 1000000
NUM_SEEDS = 5
EVAL_EPISODES = 20

# Output directories
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results_final")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints_final")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# AGENT CONFIGS - ALL WITH MlpPolicy
# ============================================================

AGENTS = {
    "PPO": {
        "name": "PPO (On-Policy)",
        "policy": "MlpPolicy",  # VECTOR OBSERVATIONS
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 3,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
    },
    "SAC": {
        "name": "SAC (Off-Policy)",
        "policy": "MlpPolicy",  # VECTOR OBSERVATIONS
        "learning_rate": 3e-4,
        "buffer_size": 10000,
        "batch_size": 64,
        "learning_starts": 1000,
        "tau": 0.005,
        "target_update_interval": 1,
        "train_freq": 1,
        "gamma": 0.99,
        "ent_coef": 0.1,
    },
    "TD3": {
        "name": "TD3 (Off-Policy)",
        "policy": "MlpPolicy",  # VECTOR OBSERVATIONS
        "learning_rate": 3e-4,
        "buffer_size": 10000,
        "batch_size": 64,
        "learning_starts": 1000,
        "tau": 0.005,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "train_freq": 1,
        "gamma": 0.99,
    }
}


def create_env(seed=None):
    """Create Reacher environment"""
    env = gym.make("Reacher-v4", render_mode="rgb_array")
    if seed is not None:
        env.reset(seed=seed)
    return env


def train_agent(agent_type, agent_config, seed=0):
    """Train a single agent"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {agent_config['name']} | Seed: {seed}")
    logger.info(f"Timesteps: {TOTAL_TIMESTEPS} | Device: {DEVICE}")
    logger.info(f"{'='*60}")

    try:
        env = create_env(seed=seed)

        # Create model based on agent type
        if agent_type == "PPO":
            model = PPO(
                policy=agent_config["policy"],
                env=env,
                learning_rate=agent_config["learning_rate"],
                n_steps=agent_config["n_steps"],
                batch_size=agent_config["batch_size"],
                n_epochs=agent_config["n_epochs"],
                gamma=agent_config["gamma"],
                gae_lambda=agent_config["gae_lambda"],
                clip_range=agent_config["clip_range"],
                vf_coef=agent_config["vf_coef"],
                ent_coef=agent_config["ent_coef"],
                verbose=0,
                device=DEVICE,
                seed=seed,
            )

        elif agent_type == "SAC":
            model = SAC(
                policy=agent_config["policy"],
                env=env,
                learning_rate=agent_config["learning_rate"],
                buffer_size=agent_config["buffer_size"],
                batch_size=agent_config["batch_size"],
                learning_starts=agent_config["learning_starts"],
                tau=agent_config["tau"],
                target_update_interval=agent_config["target_update_interval"],
                train_freq=agent_config["train_freq"],
                gamma=agent_config["gamma"],
                ent_coef=agent_config["ent_coef"],
                verbose=0,
                device=DEVICE,
                seed=seed,
            )

        elif agent_type == "TD3":
            model = TD3(
                policy=agent_config["policy"],
                env=env,
                learning_rate=agent_config["learning_rate"],
                buffer_size=agent_config["buffer_size"],
                batch_size=agent_config["batch_size"],
                learning_starts=agent_config["learning_starts"],
                tau=agent_config["tau"],
                policy_delay=agent_config["policy_delay"],
                target_policy_noise=agent_config["target_policy_noise"],
                target_noise_clip=agent_config["target_noise_clip"],
                train_freq=agent_config["train_freq"],
                gamma=agent_config["gamma"],
                verbose=0,
                device=DEVICE,
                seed=seed,
            )

        # Create checkpoint directory
        checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"{agent_type}_seed{seed}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Train
        logger.info(f"🚀 Starting training... Please wait")
        model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=50)

        # Save
        save_path = os.path.join(checkpoint_dir, "model")
        model.save(save_path)
        logger.info(f" Model saved")

        env.close()

        return {"status": "success", "model_path": save_path}

    except Exception as e:
        logger.error(f" Training failed: {str(e)}")
        return {"status": "error", "error": str(e)}


def evaluate_agent(agent_type, model_path, seed=0):
    """Evaluate trained agent"""
    try:
        env = create_env(seed=seed)

        # Load model
        if agent_type == "PPO":
            model = PPO.load(model_path, env=env)
        elif agent_type == "SAC":
            model = SAC.load(model_path, env=env)
        else:
            model = TD3.load(model_path, env=env)

        # Evaluate
        rewards = []
        for _ in range(EVAL_EPISODES):
            obs, _ = env.reset()
            ep_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                done = terminated or truncated
            rewards.append(ep_reward)

        env.close()

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        logger.info(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")

        return {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "rewards": rewards,
        }

    except Exception as e:
        logger.error(f" Evaluation failed: {str(e)}")
        return {"status": "error", "error": str(e)}


def main():
    """Main training loop"""

    print("""
╔════════════════════════════════════════════════════════════╗
║     TASK 2: DRL TRAINING - STANDALONE FINAL VERSION        ║
║         PPO vs SAC vs TD3 on MuJoCo Reacher                ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Verify GPU
    print(f"\n Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f" CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f" GPU: {torch.cuda.get_device_name(0)}")

    # Results storage
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "total_timesteps": TOTAL_TIMESTEPS,
            "num_seeds": NUM_SEEDS,
            "device": DEVICE,
        },
        "agents": {}
    }

    # Train each agent with multiple seeds
    for agent_type, agent_config in AGENTS.items():
        logger.info(f"\n{'#'*60}")
        logger.info(f"# {agent_type}: {agent_config['name']}")
        logger.info(f"{'#'*60}")

        agent_results = {"seeds": {}}

        for seed in range(NUM_SEEDS):
            # Train
            train_result = train_agent(agent_type, agent_config, seed=seed)

            if train_result["status"] == "success":
                # Evaluate
                eval_result = evaluate_agent(agent_type, train_result["model_path"], seed=seed)
                agent_results["seeds"][f"seed_{seed}"] = eval_result
                logger.info(f" Seed {seed} complete\n")
            else:
                logger.error(f" Seed {seed} failed\n")
                agent_results["seeds"][f"seed_{seed}"] = {"error": train_result["error"]}

        # Aggregate
        rewards_list = []
        for seed_result in agent_results["seeds"].values():
            if "mean_reward" in seed_result:
                rewards_list.append(seed_result["mean_reward"])

        if rewards_list:
            agent_results["aggregate"] = {
                "mean": float(np.mean(rewards_list)),
                "std": float(np.std(rewards_list)),
                "best": float(np.max(rewards_list)),
            }

        all_results["agents"][agent_type] = agent_results

    # Save results
    results_file = os.path.join(RESULTS_DIR, "results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    logger.info(f"\n Results saved to {results_file}")

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    for agent_type, agent_results in all_results["agents"].items():
        if "aggregate" in agent_results:
            agg = agent_results["aggregate"]
            print(f"\n{AGENTS[agent_type]['name']}:")
            print(f"  Mean Reward: {agg['mean']:.2f} ± {agg['std']:.2f}")
            print(f"  Best Reward: {agg['best']:.2f}")

    print("\n" + "="*60)
    print(" TASK 2 TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
