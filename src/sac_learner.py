import os

import tensorflow as tf
import numpy as np

from baselines.agent_load_manager import AgentManager
from baselines.sac.replay_buffer import ReplayBuffer
from baselines.sac.sac import SoftActorCritic


def learn(env,
          lr=1e-3,
          gamma=0.99,
          polyak=0.995,
          nb_epochs=10000,
          random_act_step = 1000,
          train_steps =100,
          batch_size = 128,
          render=False,
          log_dir='./sac_log/',
          summary_dir ='./summary/sac'):

    os.makedirs (log_dir, exist_ok=True)
    os.makedirs (summary_dir, exist_ok=True)

    writer = tf.summary.create_file_writer(summary_dir, filename_suffix=None)

    state_space = env.observation_space.shape[0]
    # TODO: fix this when env.action_space is not `Box`
    action_space = env.action_space.shape[0]

    # Initialize Replay buffer.
    replay = ReplayBuffer (state_space, action_space)

    agent = SoftActorCritic (action_space, writer,
                           learning_rate=lr,
                           gamma=gamma, polyak=polyak)

    MANAGER = AgentManager (agent, log_dir, log_dir)


    # Repeat until convergence
    global_step = 1
    episode = 1
    episode_rewards = []
    for epoch in range(nb_epochs):
        # Observe state
        current_state = env.reset ()
        done = False
        for idx in range(random_act_step):

            action = env.action_space.sample () if np.random.uniform () > 0.5 else agent.sample_action (current_state)
            current_state, reward, done, _ = env.step (action)
            end = 0 if done else 1
            replay.store (current_state, action, reward, current_state, end)


        current_state = env.reset ()
        step = 1
        episode_reward = 0
        done = False
        while not done:
            action = agent.sample_action (current_state)
            current_state, reward, done, _ = env.step(action)

            episode_reward += reward
            end = 0 if done else 1
            replay.store (current_state, action, reward, current_state, end)

            if render: env.render ()
            step += 1
            global_step += 1


        for epoch in range (train_steps):
            # Randomly sample minibatch of transitions from replay buffer
            current_states, actions, rewards, next_states, ends = replay.fetch_sample (num_samples=batch_size)
            critic1_loss, critic2_loss, actor_loss, alpha_loss = agent.train (current_states, actions, rewards, next_states, ends)

            with writer.as_default ():
                tf.summary.scalar ("actor_loss", actor_loss, agent.epoch_step)
                tf.summary.scalar ("critic1_loss", critic1_loss, agent.epoch_step)
                tf.summary.scalar ("critic2_loss", critic2_loss, agent.epoch_step)
                tf.summary.scalar ("alpha_loss", alpha_loss, agent.epoch_step)

            agent.epoch_step += 1
            agent.update_weights ()

        MANAGER.save()
        episode_rewards.append (episode_reward)
        episode += 1
        avg_episode_reward = sum (episode_rewards[-100:]) / len (episode_rewards[-100:])

        print (f"Episode {episode} reward: {episode_reward}")
        print (f"{episode} Average episode reward: {avg_episode_reward}")
        with writer.as_default ():
            tf.summary.scalar ("episode_reward", episode_reward, episode)
            tf.summary.scalar ("avg_episode_reward", avg_episode_reward, episode)
