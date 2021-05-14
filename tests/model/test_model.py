import pytest

import energy_model.model
import energy_model.mcmc.langevin
import energy_model.mcmc.mala
import energy_model.mcmc.tempered_transitions


@pytest.mark.parametrize(
    "model_factory,model_args",
    [
        (energy_model.model.SimpleEnergyModel, (3, 2, 2)),
        (energy_model.model.ConvEnergyModel, ((2, 4, 4), 2, 4)),
        (energy_model.model.ResnetEnergyModel, ((2, 4, 4), 2, 2, 4)),
    ],
)
def test_model_outputs(model_factory, model_args):
    """Tests the input and output shape of a model."""
    model = model_factory(*model_args)
    num_samples = 128
    samples = model.sample_from_prior(num_samples)
    energy = model(samples)

    assert energy.shape == (num_samples,), "Energy was not of the expected shape."


@pytest.mark.parametrize(
    "model_factory,model_args",
    [
        (energy_model.model.SimpleEnergyModel, (3, 2, 2)),
        (energy_model.model.ConvEnergyModel, ((2, 4, 4), 2, 4)),
        (energy_model.model.ResnetEnergyModel, ((2, 4, 4), 2, 2, 4)),
    ],
)
@pytest.mark.parametrize(
    "name, fsampler, num_steps",
    [
        ("langevin", lambda: energy_model.mcmc.langevin.LangevinSampler(lr=0.1), 100),
        ("mala", lambda: energy_model.mcmc.mala.MALASampler(lr=0.1), 100),
        (
            "tempered langevin",
            lambda: energy_model.mcmc.tempered_transitions.TemperedTransitions(
                mc_dynamics=energy_model.mcmc.langevin.LangevinSampler(lr=0.5)
            ),
            10,
        ),
        (
            "tempered mala",
            lambda: energy_model.mcmc.tempered_transitions.TemperedTransitions(
                mc_dynamics=energy_model.mcmc.mala.MALASampler(lr=0.5)
            ),
            10,
        ),
    ],
)
def test_model_sampling(model_factory, model_args, name, fsampler, num_steps):
    """Tests the input and output shape of a model."""
    model = model_factory(*model_args)
    num_samples = 128
    samples = model.sample_from_prior(num_samples)
    energy = model(samples)

    assert (
        samples.shape[1:] == model.input_shape
    ), "Prior samples not of the correct shape."

    assert energy.shape == (num_samples,), "Energy was not of the expected shape."

    mc_dynamics = fsampler()
    model_samples = model.sample_fantasy(
        samples, num_mc_steps=5, mc_dynamics=mc_dynamics
    )

    assert (
        model_samples.shape[1:] == model.input_shape
    ), "Model samples not of the correct shape."
