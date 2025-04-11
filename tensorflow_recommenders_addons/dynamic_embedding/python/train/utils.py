from typing import List

from tensorflow import distribute as tf_dist
"""
worker_devices property of ParameterServerStrategyV2Extended returns only /device
while ParameterServerStrategy returns /job:worker/task:0/device:CPU:0
override this property to return the same format as ParameterServerStrategy.worker_devices
"""


def worker_devices(devices: List[str], tasks: int, type: str) -> List[str]:
  if tasks % len(devices) != 0:
    raise ValueError(
        "Number of tasks must be a multiple of the number of devices.")

  return [
      f"/job:{type}/task:{i}" + devices[i % len(devices)] for i in range(tasks)
  ]


def is_parameter_server_strategy(strategy):
  return strategy is not None and isinstance(
      strategy, tf_dist.experimental.ParameterServerStrategy)
