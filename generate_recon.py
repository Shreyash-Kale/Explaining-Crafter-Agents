"""
Standalone recon-image generator.
Loads the latest checkpoint and writes a side-by-side orig/recon PNG for one
observation sampled from a fresh Crafter env. Safe to run while training is live.

Usage (from the repo root in WSL):
    python generate_recon.py --checkpoint-dir data/checkpoints/ckpt_750000 --out recon_check.png
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import imageio
import gymnasium as gym
import crafter

sys.path.insert(0, os.path.dirname(__file__))
from dreamer.core import DreamerV2


def generate(checkpoint_dir: str, out_path: str) -> None:
    env = gym.make('CrafterReward-v1')
    obs, _ = env.reset()

    agent = DreamerV2(
        observation_space=env.observation_space,
        action_space=env.action_space,
        checkpoint_dir=checkpoint_dir,
        load_checkpoint=False,  # we restore manually below
    )
    # Build weights by doing one dummy forward pass before restoring
    dummy = tf.zeros([1, *env.observation_space.shape], dtype=tf.float32)
    _ = agent.encoder(dummy)

    ckpt = tf.train.Checkpoint(
        encoder=agent.encoder,
        rssm=agent.rssm,
        decoder=agent.decoder,
    )
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    ckpt.restore(latest).expect_partial()
    print(f"Restored from {latest}")

    obs_raw = tf.cast(obs[None], tf.float32)          # [1,64,64,3]  in [0,255]
    embed = agent.encoder(obs_raw)                     # encoder normalises /255 internally
    rec_state, rec_discrete, _, _ = agent.rssm.observe(
        embed,
        tf.zeros([1, agent.action_size], dtype=tf.float32),
        *agent.rssm.initial_state(1),
    )
    feat = tf.concat([rec_state, tf.reshape(rec_discrete, [1, -1])], axis=-1)
    recon = agent.decoder(feat).mean()[0].numpy()      # (64,64,3) in [0,1]
    orig  = obs_raw[0].numpy() / 255.0                 # (64,64,3) in [0,1]

    side_by_side = np.concatenate([orig, recon], axis=1)
    imageio.imwrite(out_path, (side_by_side * 255).astype(np.uint8))
    print(f"Saved {out_path}  (left=original, right=reconstruction)")
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', default='data/checkpoints/ckpt_750000')
    parser.add_argument('--out', default='recon_check.png')
    args = parser.parse_args()
    generate(args.checkpoint_dir, args.out)
