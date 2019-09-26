import logging
import itertools
import gym
import numpy as np
from mlagents.envs import UnityEnvironment
from gym import error, spaces

from util.rendering import SimpleImageViewer

class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gym_unity")

class UnityEnvironmentWrapper(gym.Env):
    """
    Provides Gym wrapper for Unity Learning Environments.
    Multi-agent environments use lists for object types, as done here:
    https://github.com/openai/multiagent-particle-envs
    """

    def __init__(
        self,
        environment_filename: str,
        worker_id: int = 0,
        use_visual: bool = False,
        use_vector: bool = False,
        flatten_branched: bool = False,
        no_graphics: bool = False,
        allow_multiple_visual_obs: bool = False,
        docker_training: bool = False
    ):
        """
        Environment initialization
        :param environment_filename: The UnityEnvironment path or file to be wrapped in the gym.
        :param worker_id: Worker number for environment.
        :param use_visual: Whether to use visual observation or vector observation.
        :param flatten_branched: If True, turn branched discrete action spaces into a Discrete space rather than
            MultiDiscrete.
        :param no_graphics: Whether to run the Unity simulator in no-graphics mode
        :param allow_multiple_visual_obs: If True, return a list of visual observations instead of only one.
        """
        self._env = UnityEnvironment(
            environment_filename, 
            worker_id, 
            no_graphics=no_graphics, 
            docker_training=docker_training
        )
        self.name = self._env.academy_name
        self.visual_obs = None
        self.vector_obs = None
        self._flattener = None
        self._allow_multiple_visual_obs = allow_multiple_visual_obs

        # Check brain configuration
        if len(self._env.brains) != 1:
            raise UnityGymException(
                "There can only be one brain in a UnityEnvironment "
                "if it is wrapped in a gym."
            )
        if len(self._env.external_brain_names) <= 0:
            raise UnityGymException(
                "There are not any external brain in the UnityEnvironment"
            )

        self.brain_name = self._env.external_brain_names[0]
        brain = self._env.brains[self.brain_name]

        if use_visual and brain.number_visual_observations == 0:
            raise UnityGymException(
                "`use_visual` was set to True, however there are no"
                " visual observations as part of this environment."
            )
        self.use_visual = brain.number_visual_observations >= 1 and use_visual
        self.use_vector = use_vector

        if brain.number_visual_observations > 1 and not self._allow_multiple_visual_obs:
            logger.warning(
                "The environment contains more than one visual observation. "
                "You must define allow_multiple_visual_obs=True to received them all. "
                "Otherwise, please note that only the first will be provided in the observation."
            )

        if brain.num_stacked_vector_observations != 1:
            raise UnityGymException(
                "There can only be one stacked vector observation in a UnityEnvironment "
                "if it is wrapped in a gym."
            )

        # Set observation and action spaces
        if brain.vector_action_space_type == "discrete":
            if len(brain.vector_action_space_size) == 1:
                self._action_space = spaces.Discrete(brain.vector_action_space_size[0])
            else:
                if flatten_branched:
                    self._flattener = ActionFlattener(brain.vector_action_space_size)
                    self._action_space = self._flattener.action_space
                else:
                    self._action_space = spaces.MultiDiscrete(
                        brain.vector_action_space_size
                    )
        else:
            if flatten_branched:
                logger.warning(
                    "The environment has a non-discrete action space. It will "
                    "not be flattened."
                )
            high = np.array([1] * brain.vector_action_space_size[0])
            self._action_space = spaces.Box(-high, high, dtype=np.float32)
        high = np.array([np.inf] * brain.vector_observation_space_size)
        self.action_meanings = brain.vector_action_descriptions
        if self.use_visual:
            if brain.camera_resolutions[0]["blackAndWhite"]:
                depth = 1
            else:
                depth = 3
            self._observation_space = spaces.Box(
                0,
                1,
                dtype=np.float32,
                shape=(
                    brain.camera_resolutions[0]["height"],
                    brain.camera_resolutions[0]["width"],
                    depth,
                ),
            )
        else:
            self._observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        In the case of multi-agent environments, this is a list.
        Returns: observation (object/list): the initial observation of the
            space.
        """
        current_state = self._env.reset()[self.brain_name]

        obs, reward, done = self._single_step(current_state)
        return obs

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        In the case of multi-agent environments, these are lists.
        Args:
            action (object/list): an action provided by the environment
        Returns:
            observation (object/list): agent's observation of the current environment
            reward (float/list) : amount of reward returned after previous action
            done (boolean/list): whether the episode has ended.
            info (dict): contains auxiliary diagnostic information, including BrainInfo.
        """

        # Use random actions for all other agents in environment.
        if self._flattener is not None:
            # Translate action into list
            action = self._flattener.lookup_action(action)

        current_state = self._env.step(action)[self.brain_name]

        obs, reward, done = self._single_step(current_state)
        return obs, reward, done

    def _single_step(self, current_state):
        if self.use_visual:
            visual_obs = current_state.visual_observations

            if self._allow_multiple_visual_obs:
                self.visual_obs = np.dstack([obs[0] for obs in visual_obs]).astype(np.float32)
            else:
                self.visual_obs = visual_obs[0][0]

        if self.use_vector:
            self.vector_obs = current_state.vector_observations[0, :].astype(np.float32)

        return (self.visual_obs, self.vector_obs), current_state.rewards[0], current_state.local_done[0]


    def render(self, mode='rgb_array'):
        img = self.visual_obs
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._env.close()

    def get_action_meanings(self):
        return self.action_meanings

    @property
    def metadata(self):
        return {"render.modes": ["rgb_array"]}

    @property
    def reward_range(self):
        return -float("inf"), float("inf")

    @property
    def spec(self):
        return None

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def number_agents(self):
        return self._n_agents


class ActionFlattener:
    """
    Flattens branched discrete action spaces into single-branch discrete action spaces.
    """

    def __init__(self, branched_action_space):
        """
        Initialize the flattener.
        :param branched_action_space: A List containing the sizes of each branch of the action
        space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.
        """
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = spaces.Discrete(len(self.action_lookup))

    @classmethod
    def _create_lookup(self, branched_action_space):
        """
        Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
        Each key in the Dict maps to one unique set of branched actions, and each value
        contains the List of branched actions.
        """
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        # Dict should be faster than List for large action spaces
        action_lookup = {
            _scalar: _action for (_scalar, _action) in enumerate(all_actions)
        }
        return action_lookup

    def lookup_action(self, action):
        """
        Convert a scalar discrete action into a unique set of branched actions.
        :param: action: A scalar value representing one of the discrete actions.
        :return: The List containing the branched actions.
        """
        return self.action_lookup[action]
