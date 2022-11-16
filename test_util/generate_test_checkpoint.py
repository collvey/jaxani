import optax
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from neurochem.parse_resources import parse_neurochem_resources
from jaxani.constants import Constants
from jaxani.model import load_model_ensemble
from flax.training import train_state, checkpoints

INFO_FILE = 'ani-2x_8x.info'
CKPT_DIR = os.path.join(os.path.dirname(__file__), '../test/test_ckpts')
CKPT_PREFIX = f'test_ensemble_'

def generate_test_checkpoint():
  # Loads model ensemble and model params
  const_file, sae_file, ensemble_prefix, ensemble_size = parse_neurochem_resources(INFO_FILE)
  consts = Constants(const_file)
  jax_model_ensemble, jax_model_params = load_model_ensemble(consts.species, ensemble_prefix, ensemble_size)

  # Create TrainState and save checkpoint
  tx = optax.adam(learning_rate=0)
  state = train_state.TrainState.create(apply_fn=jax_model_ensemble.apply, params=jax_model_params, tx=tx)
  ckpt_path = os.path.join(CKPT_DIR, f'{CKPT_PREFIX}0')
  if not os.path.exists(ckpt_path):
    checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=0, prefix=CKPT_PREFIX)

if __name__ == '__main__':
  generate_test_checkpoint()