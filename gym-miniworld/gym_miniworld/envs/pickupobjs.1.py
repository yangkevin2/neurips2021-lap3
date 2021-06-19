import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, Ball, Key
from ..params import DEFAULT_PARAMS

class PickupObjs(MiniWorldEnv):
    """
    Room with multiple objects. The agent collects +1 reward for picking up
    each object. Objects disappear when picked up.
    """

    def __init__(self, size=12, num_objs=2, **kwargs):
        assert size >= 2
        self.size = size
        self.num_objs = num_objs

        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', 0.05)

        super().__init__(
            max_episode_steps=200,
            params=params,
            **kwargs
        )

        # Reduce the action space
        self.action_space = spaces.Discrete(self.actions.pickup+1)

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex='brick_wall',
            floor_tex='asphalt',
            no_ceiling=True,
        )

        self.place_agent(min_x=self.size/2-0.1, max_x=self.size/2+0.1, min_z=self.size/2-0.1, max_z=self.size/2+0.1)
        dist_ranges = [(4, 4.5), (5, 5.5)]
        # dist_ranges = [(3, 4), (5, 6)]

        acceptable = False
        while not acceptable:
            self.boxes = []
            for obj in range(self.num_objs):
                distance_min, distance_max = dist_ranges[obj]
                color = self.rand.color()
                box = self.place_entity(Box(color=color, size=0.9))
                box_dist = np.linalg.norm(self.agent.pos - box.pos) # higher reward for farther box
                attempts = 0
                boxes_successful = True
                while box_dist < distance_min or box_dist > distance_max:
                    attempts += 1
                    self.entities = self.entities[:-1] # re-place the box
                    box = self.place_entity(Box(color=color, size=0.9))
                    box_dist = np.linalg.norm(self.agent.pos - box.pos) # higher reward for farther box
                    if attempts > 20:
                        boxes_successful = False
                        break
                self.boxes.append((box, box_dist))
                # if obj_type == Ball:
                #     self.place_entity(Ball(color=color, size=0.9))
                # if obj_type == Key:
                #     self.place_entity(Key(color=color))
            self.boxes = sorted(self.boxes, key=lambda tup: tup[1]) # sort in increasing order of dist from agent starting pos
            self.box_dist = np.linalg.norm(self.boxes[0][0].pos - self.boxes[1][0].pos)
            # acceptable = self.box_dist > 4 and self.box_dist < 5
            acceptable = boxes_successful and self.box_dist > 4 and self.box_dist < 5
            if not acceptable:
                self.entities = self.entities[:-self.num_objs]

        # print('box dist', self.box_dist)

        self.num_picked_up = 0

    # def step(self, action):
    #     obs, reward, done, info = super().step(action)

    #     if self.agent.carrying:
    #         self.entities.remove(self.agent.carrying)
    #         self.agent.carrying = None
    #         self.num_picked_up += 1
    #         reward = 1

    #         if self.num_picked_up == self.num_objs:
    #             done = True

    #     return obs, reward, done, info

    def step(self, action, print_eval=False):
        obs, reward, done, info = super().step(action)

        reward = 0
        for i, (box, _) in enumerate(self.boxes):
            if self.near(box):
                # reward += 0.1*self._reward() + i # higher reward for going to a farther box
                # reward += 0.01*self._reward() + i # higher reward for going to a farther box
                reward += i
                done = True
                break
        
        # Added dense reward
        # if reward == 0 and done:
        if done:
            for box, _ in self.boxes:
                reward += -math.sqrt(np.linalg.norm(self.agent.pos - box.pos)) # sum of distances to boxes
            reward += math.sqrt(self.box_dist) # baseline distance between boxes

        return obs, reward, done, info
