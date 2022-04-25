import os
import numpy as np
import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env
import  time



class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, mass_scale_set=[0.85, 0.9, 0.95, 1.0], damping_scale_set=[1.0]):
        mujoco_env.MujocoEnv.__init__(self, "swimmer.xml", 4)
        utils.EzPickle.__init__(self)

        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)

        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set
        self.label_index = None
        self.proc_observation_space_dims = self.obs_preproc(self._get_obs()).shape[-1]
        #  self._set_observation_space(self._get_obs())
        utils.EzPickle.__init__(self, mass_scale_set, damping_scale_set)

    def _set_observation_space(self, observation):
        super(SwimmerEnv, self)._set_observation_space(observation)
        proc_observation = self.obs_preproc(observation[None])
        self.proc_observation_space_dims = proc_observation.shape[-1]

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        self.xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - self.xposbefore) / self.dt
        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([
            ((self.sim.data.qpos[0] - self.xposbefore) / self.dt).flat,
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])
            # np.concatenate([qpos.flat[2:], qvel.flat])

    def get_labels(self):
        return self.label_index

    def obs_preproc(self, obs):
        return obs

    def obs_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv),
        )

        try:
            self.reset_num = int(str(time.time())[-2:])
        except:
            self.reset_num = 1
        self.np_random.seed(self.reset_num)
        self.xposbefore = self.sim.data.qpos[0]
        random_index = self.np_random.randint(len(self.mass_scale_set))
        self.mass_scale = self.mass_scale_set[random_index]
        self.label_index = random_index * len(self.damping_scale_set)
        random_index = self.np_random.randint(len(self.damping_scale_set))
        self.damping_scale = self.damping_scale_set[random_index]
        self.label_index = random_index + self.label_index
        self.change_env()
        return self._get_obs()

    def change_env(self):
        mass = np.copy(self.original_mass)
        damping = np.copy(self.original_damping)
        mass *= self.mass_scale
        damping *= self.damping_scale

        self.model.body_mass[:] = mass
        self.model.dof_damping[:] = damping

    def reward(self, obs, act, next_obs):
        reward_ctrl = np.sum(np.square(act), axis=-1)
        reward_run = obs[..., 0]

        # reward_contact = 0.0
        # reward_survive = 0.05
        reward = reward_run - 0.0001*reward_ctrl

        return reward

    def tf_reward_fn(self):
        def _thunk(obs, act, next_obs):
            reward_ctrl = -0.0001 * tf.reduce_sum(tf.square(act), axis=-1)
            reward_run = obs[..., 0]

            # reward_contact = 0.0
            # reward_survive = 0.05
            reward = reward_run + reward_ctrl
            return reward
        return _thunk

    def seed(self, seed=None):
        if seed is None:
            self._seed = 0
        else:
            self._seed = seed
        super().seed(seed)

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_sim_parameters(self):
        return np.array([self.mass_scale, self.damping_scale])

    def num_modifiable_parameters(self):
        return 2

    def log_diagnostics(self, paths, prefix):
        return

