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

"""Super simple checkpoints using numpy."""

import dataclasses
import datetime
import os
from typing import Optional
import zipfile

from absl import logging
from ferminet import networks
from ferminet import observables
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np


def _broadcast_str(value: Optional[str]) -> Optional[str]:
  """Broadcasts an optional string from host 0 to all hosts."""
  if jax.process_count() == 1:
    return value
  encoded = (value or '').encode('utf-8')
  length = int(
      multihost_utils.broadcast_one_to_all(
          jnp.asarray([len(encoded)], dtype=jnp.int32)
      )[0]
  )
  if length == 0:
    return None
  if jax.process_index() == 0:
    buf = np.frombuffer(encoded, dtype=np.uint8)
  else:
    buf = np.zeros(length, dtype=np.uint8)
  buf = multihost_utils.broadcast_one_to_all(buf)
  return bytes(buf).decode('utf-8')


def find_last_checkpoint(ckpt_path: Optional[str] = None) -> Optional[str]:
  """Finds most recent valid checkpoint in a directory.

  Args:
    ckpt_path: Directory containing checkpoints.

  Returns:
    Last QMC checkpoint (ordered by sorting all checkpoints by name in reverse)
    or None if no valid checkpoint is found or ckpt_path is not given or doesn't
    exist. A checkpoint is regarded as not valid if it cannot be read
    successfully using np.load.
  """
  chosen = None
  if jax.process_index() == 0:
    if ckpt_path and os.path.exists(ckpt_path):
      files = [
          f
          for f in os.listdir(ckpt_path)
          if 'qmcjax_ckpt_' in f and f.endswith('.npz')
      ]
      # Handle case where last checkpoint is corrupt/empty.
      for file in sorted(files, reverse=True):
        fname = os.path.join(ckpt_path, file)
        with open(fname, 'rb') as f:
          try:
            np.load(f, allow_pickle=True)
            chosen = fname
            break
          except (OSError, EOFError, zipfile.BadZipFile):
            logging.info(
                'Error loading checkpoint %s. Trying next checkpoint...', fname
            )
  return _broadcast_str(chosen)


def create_save_path(save_path: Optional[str]) -> str:
  """Creates the directory for saving checkpoints, if it doesn't exist.

  Args:
    save_path: directory to use. If false, create a directory in the working
      directory based upon the current time.

  Returns:
    Path to save checkpoints to.
  """
  timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
  default_save_path = os.path.join(os.getcwd(), f'ferminet_{timestamp}')
  ckpt_save_path = _broadcast_str(save_path or default_save_path)
  assert ckpt_save_path is not None
  if jax.process_index() == 0 and not os.path.isdir(ckpt_save_path):
    os.makedirs(ckpt_save_path, exist_ok=True)
  if jax.process_count() > 1:
    multihost_utils.sync_global_devices('checkpoint_save_path_ready')
  return ckpt_save_path


def get_restore_path(restore_path: Optional[str] = None) -> Optional[str]:
  """Gets the path containing checkpoints from a previous calculation.

  Args:
    restore_path: path to checkpoints.

  Returns:
    The path or None if restore_path is falsy.
  """
  if restore_path:
    ckpt_restore_path = restore_path
  else:
    ckpt_restore_path = None
  return ckpt_restore_path


def save(save_path: str,
         t: int,
         data: networks.FermiNetData,
         params,
         opt_state,
         mcmc_width,
         density_state: Optional[observables.DensityState] = None) -> str:
  """Saves checkpoint information to a npz file.

  Args:
    save_path: path to directory to save checkpoint to. The checkpoint file is
      save_path/qmcjax_ckpt_$t.npz, where $t is the number of completed
      iterations.
    t: number of completed iterations.
    data: MCMC walker configurations.
    params: pytree of network parameters.
    opt_state: optimization state.
    mcmc_width: width to use in the MCMC proposal distribution.
    density_state: optional state of the density matrix calculation

  Returns:
    path to checkpoint file.
  """
  ckpt_filename = os.path.join(save_path, f'qmcjax_ckpt_{t:06d}.npz')
  tmp_filename = f'{ckpt_filename}.tmp.{os.getpid()}'
  logging.info('Saving checkpoint %s', ckpt_filename)
  with open(tmp_filename, 'wb') as f:
    np.savez(
        f,
        t=t,
        data=dataclasses.asdict(data),  # pyrefly: ignore[bad-argument-type]
        params=params,
        opt_state=np.asarray(opt_state, dtype=object),
        mcmc_width=mcmc_width,
        density_state=(dataclasses.asdict(density_state)  # pyrefly: ignore[bad-argument-type]
                       if density_state else None))
  os.replace(tmp_filename, ckpt_filename)
  return ckpt_filename


def restore(restore_filename: str, batch_size: Optional[int] = None):
  """Restores data saved in a checkpoint.

  Args:
    restore_filename: filename containing checkpoint.
    batch_size: total batch size to be used. If present, check the data saved in
      the checkpoint is consistent with the batch size requested for the
      calculation.

  Returns:
    (t, data, params, opt_state, mcmc_width) tuple, where
    t: number of completed iterations.
    data: MCMC walker configurations.
    params: pytree of network parameters.
    opt_state: optimization state.
    mcmc_width: width to use in the MCMC proposal distribution.
    density_state: optional state of the density matrix calculation

  Raises:
    ValueError: if the leading dimension of data does not match the number of
    devices (i.e. the number of devices being parallelised over has changed) or
    if the total batch size is not equal to the number of MCMC configurations in
    data.
  """
  logging.info('Loading checkpoint %s', restore_filename)
  with open(restore_filename, 'rb') as f:
    ckpt_data = np.load(f, allow_pickle=True)
    # Retrieve data from npz file. Non-array variables need to be converted back
    # to natives types using .tolist().
    t = ckpt_data['t'].tolist() + 1  # Return the iterations completed.
    data = networks.FermiNetData(**ckpt_data['data'].item())
    params = ckpt_data['params'].tolist()
    opt_state = ckpt_data['opt_state'].tolist()
    mcmc_width = jnp.array(ckpt_data['mcmc_width'].tolist())
    if ckpt_data['density_state']:
      density_state = observables.DensityState(
          **ckpt_data['density_state'].item())
    else:
      density_state = None
    total_devices = jax.device_count()
    local_devices = jax.local_device_count()
    ckpt_local_devices = int(mcmc_width.shape[0])
    if ckpt_local_devices != local_devices:
      raise ValueError(
          f'Checkpoint was saved with {ckpt_local_devices} local devices per '
          f'host, but current process has {local_devices} local devices.'
      )
    total_walkers = data.positions.shape[0] * data.positions.shape[1]
    if batch_size and total_walkers != batch_size:
      raise ValueError(
          f'Wrong batch size in loaded data. Expected {batch_size}, found '
          f'{total_walkers}.')
    if total_walkers % total_devices != 0:
      raise ValueError(
          f'Total walkers {total_walkers} not divisible by '
          f'{total_devices} total devices.'
      )
    host_idx = jax.process_index()
    start = host_idx * local_devices
    end = start + local_devices

    def _slice_host(x):
      x = np.asarray(x)
      per_device = (x.shape[0] * x.shape[1]) // total_devices
      x = x.reshape((total_devices, per_device) + x.shape[2:])
      return x[start:end]

    data = networks.FermiNetData(
        positions=_slice_host(data.positions),
        spins=_slice_host(data.spins),
        atoms=_slice_host(data.atoms),
        charges=_slice_host(data.charges),
    )
    if density_state is not None:
      density_state = observables.DensityState(
          t=density_state.t,
          positions=_slice_host(density_state.positions),
          probabilities=_slice_host(density_state.probabilities),
          move_width=density_state.move_width,
          pmove=density_state.pmove,
          mo_coeff=density_state.mo_coeff,
      )
  return t, data, params, opt_state, mcmc_width, density_state
