# content of test_tmp_path.py
import json
import numbers
import os

import numpy as np
import torch

from energy_model.distributions.core import Normal
from energy_model.hparam_sweep import get_energy_trainer
from energy_model.model import SimpleEnergyModel


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.round(3).tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().round(3).tolist()
        if isinstance(obj, numbers.Number):
            return np.round(obj, 3).tolist()[0]
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def test_save_and_restore(tmp_path):
    """Test that save and restore works."""
    # Setup the trainer.
    def setup_1d(**kwargs):
        dist = Normal(np.array([-3]))
        net = SimpleEnergyModel(1, 2, 2)
        return dist, net

    trainer = get_energy_trainer(setup_1d)({"sample_size": 10, "batch_size": 2})
    # run one training step
    trainer.train()
    # save it.
    ckpt_fn = trainer._save(tmp_checkpoint_dir=str(tmp_path))
    assert os.path.dirname(ckpt_fn) == str(tmp_path), "The model should be saved at the given path."
    attributes_to_check = {
        "global_step_": lambda m: m.global_step_,
        "epoch_": lambda m: m.epoch_,
        "net_": lambda m: json.dumps(m.net_.state_dict(), cls=NumpyEncoder),
        "model_samples_": lambda m: json.dumps(
            list(m.model_samples_), cls=NumpyEncoder
        ),
        "optimizer_": lambda m: json.dumps(m.optimizer_.state_dict(), cls=NumpyEncoder),
        "sampler_state_": lambda m: m.sampler.state_dict(),
        "ais_state_": lambda m: m.ais_loss.state_dict(),
        "replay_prob": lambda m: m.replay_prob,
        "max_replay": lambda m: m.max_replay,
    }
    invariant = {"ais_state_", "replay_prob", "max_replay"}
    # Store the state;
    original_state = {k: f(trainer) for k, f in attributes_to_check.items()}

    # run some more training steps;
    for _ in range(3):
        trainer.train()

    # Store the state;
    new_state = {k: f(trainer) for k, f in attributes_to_check.items()}

    # Check that the new state differs from the stored state:
    for k in set(original_state).difference(invariant):
        assert original_state[k] != new_state[k], f"The new {k} matches"

    # Restore from the saved file:
    trainer._restore(ckpt_fn)

    restored_state = {k: f(trainer) for k, f in attributes_to_check.items()}

    assert restored_state == original_state, "The restored state differs."


def test_save_restore_ray():
    import ray
    ray.init()
    def setup_1d(**kwargs):
        dist = Normal(np.array([-3]))
        net = SimpleEnergyModel(1, 2, 2)
        return dist, net

    trainer = get_energy_trainer(setup_1d)
    from ray.tune.utils import validate_save_restore
    validate_save_restore(trainer)
    validate_save_restore(trainer, use_object_store=True)