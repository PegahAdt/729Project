"""
Environment builder for policy gradient project.

This module reads an environment configuration file (YAML) and constructs
the appropriate environment based on the `env_name` specified in that file.

Currently only Gymnasium environments are supported via the `GymEnv` wrapper
defined in `envs/gym_env.py`.  If you wish to add support for other
environment families (e.g., DeepMind Control Suite or Atari), you can add
additional imports and `startswith()` cases below.

Example YAML snippet:

```
env_name: "gym:CartPole-v1"
time_limit: 1000
visualize: false
```

Note: The `env_name` must start with `gym:` to use the Gym loader.
"""

import yaml

from envs.gym_env import GymEnv


def build_env(env_file: str, device: str, visualize: bool):
    """Build and return an environment instance.

    Args:
        env_file: Path to a YAML file containing environment configuration.
        device:  The torch device (e.g., 'cpu' or 'cuda') to which tensors will be moved.
        visualize: Whether to enable environment rendering (if supported).

    Returns:
        An instance of a subclass of BaseEnv.

    Raises:
        AssertionError: If the `env_name` in the config is not supported.
    """
    env_config = load_env_file(env_file)
    env_name = env_config["env_name"]
    print(f"Building {env_name} env")

    # Support Gym environments when env_name starts with "gym:"
    if env_name.startswith("gym:"):
        return GymEnv(env_config, device, visualize)

    # If other prefixes are needed, add more `elif` branches here.
    raise AssertionError(f"Unsupported env: {env_name}")


def load_env_file(file: str) -> dict:
    """Load a YAML configuration file for the environment.

    Args:
        file: Path to the YAML file.

    Returns:
        A dictionary containing the environment configuration.
    """
    with open(file, "r") as stream:
        return yaml.safe_load(stream)