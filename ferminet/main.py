# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main wrapper for FermiNet in JAX."""

import os
from absl import app
from absl import flags
from absl import logging
from ferminet import base_config
from ferminet import train
import jax
from ml_collections.config_flags import config_flags

# internal imports

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'Path to config file.')


def _init_distributed():
  num_tasks = int(os.environ.get('SLURM_NTASKS', '1'))
  if num_tasks <= 1:
    return

  local_device_ids = None
  env_local_ids = os.environ.get('JAX_LOCAL_DEVICE_IDS')
  if env_local_ids:
    local_device_ids = [int(x) for x in env_local_ids.split(',') if x.strip()]
  elif int(os.environ.get('SLURM_NTASKS_PER_NODE', '1')) == 1:
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible:
      local_device_ids = list(
          range(len([x for x in visible.split(',') if x.strip()]))
      )
    elif 'SLURM_GPUS_ON_NODE' in os.environ:
      local_device_ids = list(range(int(os.environ['SLURM_GPUS_ON_NODE'])))

  jax.distributed.initialize(local_device_ids=local_device_ids)

  if jax.process_count() != num_tasks:
    raise RuntimeError(
        f'Expected {num_tasks} JAX processes from SLURM_NTASKS, '
        f'got {jax.process_count()}.'
    )
  if (
      local_device_ids is not None
      and jax.local_device_count() != len(local_device_ids)
  ):
    raise RuntimeError(
        f'Expected {len(local_device_ids)} local GPUs per process, '
        f'got {jax.local_device_count()}.'
    )


def main(_):
  if jax.process_index() != 0:
    logging.set_verbosity(logging.WARNING)
  cfg = FLAGS.config
  cfg = base_config.resolve(cfg)
  if jax.process_index() == 0:
    logging.info('System config:\n\n%s', cfg)
  train.train(cfg)


def main_wrapper():
  # For calling from setuptools' console_script entry-point.
  # Initialize distributed JAX before absl parses flags / loads config files.
  _init_distributed()
  app.run(main)


if __name__ == '__main__':
  main_wrapper()
