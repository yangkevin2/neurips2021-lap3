#!/usr/bin/env python3

"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
import math
from pyglet.window import key
from pyglet import clock
import numpy as np
import gym
import gym_miniworld

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Hallway-v0')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
parser.add_argument('--top_view', action='store_true', help='show the top view instead of the agent view')
parser.add_argument('--seed', type=int, default=123, help='seed for eval')
args = parser.parse_args()

env = gym.make(args.env_name)
env.seed(args.seed)

if args.no_time_limit:
    env.max_episode_steps = math.inf
if args.domain_rand:
    env.domain_rand = True

view_mode = 'top' if args.top_view else 'agent'

env.reset()

# Create the display window
env.render('pyglet', view=view_mode)

def step(action):
    print('step {}/{}: {}'.format(env.step_count+1, env.max_episode_steps))

    obs, reward, done, info = env.step(action)

    if reward > 0:
        print('reward={:.2f}'.format(reward))

    if done:
        print('done!')
        env.reset()

    env.render('pyglet', view=view_mode)

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render('pyglet', view=view_mode)
        return

    if symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    if symbol == key.UP:
        # step(env.actions.move_forward)
        step([1000, 0])
    elif symbol == key.DOWN:
        # step(env.actions.move_back)
        step([-1000, 0])

    elif symbol == key.LEFT:
        # step(env.actions.turn_left)
        step([0, -1000])
    elif symbol == key.RIGHT:
        # step(env.actions.turn_right)
        step([0, 1000])

    elif symbol == key.PAGEUP or symbol == key.P:
        step(env.actions.pickup)
    elif symbol == key.PAGEDOWN or symbol == key.D:
        step(env.actions.drop)

    elif symbol == key.ENTER:
        step(env.actions.done)

@env.unwrapped.window.event
def on_key_release(symbol, modifiers):
    pass

@env.unwrapped.window.event
def on_draw():
    env.render('pyglet', view=view_mode)

@env.unwrapped.window.event
def on_close():
    pyglet.app.exit()

# Enter main event loop
pyglet.app.run()

env.close()
