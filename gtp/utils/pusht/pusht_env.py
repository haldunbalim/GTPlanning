import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from gymnasium.envs import register
from . import DrawOptions

def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom


class PushTRealEnv(gym.Env):
    metadata = {"render.modes": [
        "rgb_array", "human"], "video.frames_per_second": 10}
    shapes = ['tee', 'cee', 'lee']
    masks = {
        'tee': [0.0, 0.5, 0.5, 0.0],
        'lee': [1.0, 0.0, 0.0, 0.0],
        'cee': [1.0, 0.0, 0.0, 1.0],
    }


    def __init__(self,
                 legacy=False,
                 block_cog=None, damping=None,
                 render_action=True,
                 render_size=96,
                 reset_to_state=None,
                 render_mode="rgb_array",
                 reward_type="dense",
                 ):
        self.window_size = 512  # The size of the PyGame window
        self.rela_pos_scale = self.window_size / 4
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 350, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.legacy = legacy
        self.metadata['render_mode'] = [render_mode]
        assert reward_type in ["dense", "coverage", "sparse"], "Unsupported reward type"
        self.reward_type = reward_type

        self.observation_space = spaces.Box(
            low=np.array([-2, -2, -2, -2, -1, -1,], dtype=np.float32),
            high=np.array([2, 2, 2, 2, 1, 1,], dtype=np.float32),
            shape=(6,),
            dtype=np.float32
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

        # no curriculum_leanring in base env
        self.curriculum_level=1.0
        self.random_init_pose=True
        self.agent_hits_wall = False
        
        
    def reset(self, seed=None, options=None):
        if not self.random_init_pose:
            rs = np.random.RandomState(seed=0)
        elif seed is not None:
            seed = seed
            rs = np.random.RandomState(seed=seed)
        else:
            rs = np.random.RandomState()
        self.best_coverage = 0
        self.shape_type = 'tee'
        if options is not None and 'shape_type' in options:
            self.shape_type = options['shape_type']
        # use legacy RandomState for compatibility

        self.goal_pose = np.array([256, 256, 0.]) # np.pi/4
        # self.goal_pose = np.array([342, 270, 2.1])
        self._setup(shape_type=self.shape_type)
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        state = self.reset_to_state
        if state is None:
            lower = int(220 - 120)
            upper = int(280 + 120)
            state = np.array([
                rs.randint(lower, upper),  rs.randint(lower, upper),
                rs.randint(lower, upper), rs.randint(lower, upper),
                (rs.rand() * 2 - 1) * np.pi
            ])
        self._set_state(state)

        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        dt = 1.0 / self.sim_hz
        action = np.clip(action, -1, 1)
        self.n_contact_points = 0
        self.agent_touches_block = False
        self.block_touches_wall = False
        self.agent_hits_wall = False
        n_steps = self.sim_hz // self.control_hz
        pre_block_pos = np.array(self.block.position)
        action_diff = 0
        if self.latest_action is not None and action is not None:
            action_diff = action - self.latest_action
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                desired_velocity = Vec2d(*(self.k_p * action))
                reached_position, hit_wall = self._resolve_agent_target_position(
                    desired_velocity, dt
                )
                self.agent_hits_wall = self.agent_hits_wall or hit_wall
                self.agent.velocity = (reached_position - self.agent.position) / dt

                # Step physics.
                self.space.step(dt)
                self.agent.position = reached_position
                if hit_wall:
                    self.agent.velocity = Vec2d(0, 0)
        post_block_pos = np.array(self.block.position)

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        def angle_normalize(x):
            return ((x + np.pi) % (2 * np.pi)) - np.pi

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area

        def body_point_dist(body, point):
            return min([sh.point_query(point).distance for sh in body.shapes])

        angle_diff = (angle_normalize(
            self.block.angle - self.goal_pose[2]) / np.pi) ** 2
        pos_diff = np.sum(
            np.array(((self.block.position - self.goal_pose[:2]) / self.window_size)) ** 2)
        dist_agent_block = (body_point_dist(
            self.block, self.agent.position) / self.window_size) ** 2
        agent_outside_map = self.agent.position.x < 0 or self.agent.position.y < 0 \
                            or self.agent.position.x > self.window_size or self.agent.position.y > self.window_size
        block_moved = np.sum(np.abs(pre_block_pos - post_block_pos)) > 0

        if self.reward_type == "dense":
            reward = 0
            reward -= angle_diff + pos_diff
            #reward -= (not self.agent_touches_block) * 2.0
            reward -= dist_agent_block * 1.0
            reward -= (not block_moved) * 0.25
            reward -= self.block_touches_wall * 1.0
            reward -= agent_outside_map * 0.25
            if done := np.clip(coverage / self.success_threshold, 0, 1) > self.success_threshold:
                reward = 0
        elif self.reward_type == "coverage":
            reward = coverage-1
            if done := np.clip(coverage / self.success_threshold, 0, 1) > self.success_threshold:
                reward += 10
        elif self.reward_type == "sparse":
            reward = 1.0 if coverage > self.success_threshold else 0.0

        observation = self._get_obs()
        info = self._get_info()
        info['pos_diff'] = pos_diff
        info['angle_diff'] = angle_diff
        info['dist_agent_block'] = dist_agent_block
        info['block_moved'] = block_moved
        # a python bool: gymnasium's vector info merge keeps numpy.bool_ values in an object array
        info['is_success'] = bool(done)

        return observation, reward, done, False, info

    def render(self, waypoints=None):
        return self._render_frame(self.metadata['render.modes'][0], waypoints=waypoints)

    def _get_obs(self):
        def normalize_coord(coord):
            return np.clip(np.array([coord.x/self.window_size, coord.y/self.window_size])*2-1, -2, 2)
        obs = np.array(
            tuple(normalize_coord(self.agent.position))
            + tuple(normalize_coord(self.block.position))
            + (np.cos(self.block.angle), np.sin(self.block.angle)))
        return obs.astype(np.float32)

        # based on block frame
        agent_rel_pos = self.block.world_to_local((self.agent.position[0], self.agent.position[1]))
        goal_real_pos = self.block.world_to_local((self.goal_pose[0], self.goal_pose[1]))
        goal_real_angle = self.goal_pose[2] - self.block.angle
        obs = np.array([agent_rel_pos[0] / self.rela_pos_scale, agent_rel_pos[1] / self.rela_pos_scale,
                        goal_real_pos[0] / self.rela_pos_scale, goal_real_pos[1] / self.rela_pos_scale,
                        np.sin(goal_real_angle), np.cos(goal_real_angle)])
        #obs = np.concatenate([obs, self.masks[self.shape_type]])
        return obs.astype(np.float32)

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(
            np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step,
            'agent_touches_block': self.agent_touches_block,
            'agent_hits_wall': self.agent_hits_wall
        }
        return info

    def _get_agent_position_bounds(self):
        wall_x = []
        wall_y = []
        wall_radius = 0.0
        for wall in self.walls:
            wall_x.extend([wall.a.x, wall.b.x])
            wall_y.extend([wall.a.y, wall.b.y])
            wall_radius = max(wall_radius, wall.radius)

        min_x = min(wall_x) + wall_radius + self.agent_radius
        max_x = max(wall_x) - wall_radius - self.agent_radius
        min_y = min(wall_y) + wall_radius + self.agent_radius
        max_y = max(wall_y) - wall_radius - self.agent_radius
        return min_x, max_x, min_y, max_y

    def _clamp_agent_position(self, position):
        min_x, max_x, min_y, max_y = self._get_agent_position_bounds()
        return Vec2d(
            np.clip(position.x, min_x, max_x),
            np.clip(position.y, min_y, max_y)
        )

    def _resolve_agent_target_position(self, velocity, dt):
        current_position = self._clamp_agent_position(self.agent.position)
        self.agent.position = current_position
        delta = velocity * dt
        target_position = current_position + delta
        min_x, max_x, min_y, max_y = self._get_agent_position_bounds()

        hit_fractions = [1.0]
        if target_position.x < min_x and delta.x < 0:
            hit_fractions.append((min_x - current_position.x) / delta.x)
        elif target_position.x > max_x and delta.x > 0:
            hit_fractions.append((max_x - current_position.x) / delta.x)

        if target_position.y < min_y and delta.y < 0:
            hit_fractions.append((min_y - current_position.y) / delta.y)
        elif target_position.y > max_y and delta.y > 0:
            hit_fractions.append((max_y - current_position.y) / delta.y)

        movement_fraction = float(np.clip(min(hit_fractions), 0.0, 1.0))
        reached_position = current_position + delta * movement_fraction
        hit_wall = movement_fraction < 1.0
        return reached_position, hit_wall

    def _render_frame(self, mode, waypoints=None):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(
                v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        pygame.draw.circle(canvas, self.debugging_color, self.goal_pose[:-1], 10)

        # Draw waypoints (purple, fading alpha from 0->N).
        if waypoints is not None:
            wp = np.asarray(waypoints, dtype=np.float32) * self.window_size / 2 + self.window_size / 2
            if wp.ndim == 2 and wp.shape[1] == 2 and len(wp) > 0:
                N = len(wp)

                # Draw onto a transparent overlay so per-point alpha works.
                overlay = pygame.Surface(
                    (self.window_size, self.window_size), pygame.SRCALPHA)

                # Radius scales with render size (roughly consistent visually).
                radius = max(2, int(6 * (self.window_size / 512.0)))

                for i, (x, y) in enumerate(wp):
                    # Clamp to canvas bounds
                    x = float(np.clip(x, 0, self.window_size - 1))
                    y = float(np.clip(y, 0, self.window_size - 1))

                    # alpha: strong at i=0, fades towards i=N-1
                    if N == 1:
                        alpha = 255
                    else:
                        alpha = int(
                            np.clip(255 * (1.0 - (i / (N - 1))), 0, 255))

                    pygame.draw.circle(
                        overlay, (128, 0, 128, alpha), (int(x), int(y)), radius)

                canvas.blit(overlay, (0, 0))

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"

        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                               color=(255, 0, 0), markerType=cv2.MARKER_CROSS,
                               markerSize=marker_size, thickness=thickness)
        return img

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _handle_collision(self, arbiter, space, data):
        shp1, shp2 = arbiter.shapes
        body1, body2 = shp1.body, shp2.body
        if (body1 == self.agent and body2 == self.block) or (body1 == self.block and body2 == self.agent):
            self.agent_touches_block = True
        if (isinstance(shp1, pymunk.shapes.Segment) and body2 == self.block) or \
                (isinstance(shp2, pymunk.shapes.Segment) and body1 == self.block):
            self.block_touches_wall = True
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        self.agent.position = self._clamp_agent_position(self.agent.position)
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2],
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation)
            + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self, shape_type):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.walls = walls
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 7.5)
        self.agent_radius = next(iter(self.agent.shapes)).radius
        if shape_type == 'tee':
            self.block = self.add_tee((256, 300), 0)
        elif shape_type == 'cee':
            self.block = self.add_cee((256, 300), 0)
        elif shape_type == 'lee':
            self.block = self.add_lee((256, 300), 0)
        else:
            raise ValueError(f'Unsupported shape {shape_type}')
        self.goal_color = pygame.Color('LightGreen')
        self.debugging_color = pygame.Color('red')

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0
        self.agent_touches_block = False
        self.block_touches_wall = False

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        # https://htmlcolorcodes.com/color-names
        shape.color = pygame.Color('LightGray')
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_shape(self, mass_ls, vertices_ls, position, angle, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        from functools import reduce
        inertias = [pymunk.moment_for_poly(
            mass, vertices=v) for mass, v in zip(mass_ls, vertices_ls)]
        body = pymunk.Body(sum(mass_ls), sum(inertias))
        shapes = []
        for v in vertices_ls:
            shape = pymunk.Poly(body, v)
            shape.color = pygame.Color(color)
            shape.filter = pymunk.ShapeFilter(mask=mask)
            shapes.append(shape)
        body.center_of_gravity = (reduce(
            lambda x, y: x+y, [shape.center_of_gravity for shape in shapes])) / len(shapes)
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, *shapes)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        length = 4
        length *= scale

        vertices1 = [(-length/2, scale),
                     (length/2, scale),
                     (length/2, 0),
                     (-length/2, 0)]
        vertices2 = [(-scale/2, scale),
                     (-scale/2, length),
                     (scale/2, length),
                     (scale/2, scale)]
        vertices_ls = [vertices1, vertices2]
        return self.add_shape([0.5, 0.5], vertices_ls, position, angle, color, mask)

    def add_cee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        length = 4
        length *= scale
        vertices1 = [(-length / 2, scale),
                     (length / 2, scale),
                     (length / 2, 0),
                     (-length / 2, 0)]

        vertices2 = [(-length / 2, scale),
                     (-length / 2, length),
                     (-length / 2 + scale, length),
                     (-length / 2 + scale, scale)]

        vertices3 = [(length / 2 - scale, scale),
                     (length / 2 - scale, length),
                     (length / 2, length),
                     (length / 2, scale)]
        vertices_ls = [vertices1, vertices2, vertices3]
        return self.add_shape([0.5, 0.25, 0.5], vertices_ls, position, angle, color, mask)

    def add_lee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        length = 4
        length *= scale

        # vertices1 = [(-length/3, -length/6),
        #              (length*2/3, -length/6),
        #              (length*2/3, -length/2),
        #              (-length/3, -length/2)]
        # vertices2 = [(-length/3, -length/6),
        #              (-length/3, length/2),
        #              (0, length/2),
        #              (0, -length/6)]
        vertices1 = [(-length / 2, scale),
                     (length / 2, scale),
                     (length / 2, 0),
                     (-length / 2, 0)]
        vertices2 = [(-length / 2, scale),
                     (-length / 2, length),
                     (-length / 2 + scale, length),
                     (-length / 2 + scale, scale)]
        vertices_ls = [vertices1, vertices2]
        return self.add_shape([0.5, 0.25], vertices_ls, position, angle, color, mask)


register(
    id = 'PushT-v0',
    entry_point='gtp.utils.pusht.pusht_env:PushTRealEnv',
    max_episode_steps = 300
)

if __name__ == "__main__":
    env = PushTRealEnv()
    print(env.reset())
