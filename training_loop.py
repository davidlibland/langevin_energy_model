import random
from collections import deque
from typing import List

import torch
import torch.optim as optim
import torch.utils.data as data

from model import BaseEnergyModel

# Globals
from utils.logging import tb_writer

MAX_REPLAY = 1000
REPLAY_PROB = .95


class CheckpointCallback:
    def __call__(self, net: BaseEnergyModel, data_sample, model_sample, global_step, epoch, validation: bool=False):
        raise NotImplementedError


def train(net: BaseEnergyModel, dataset: data.Dataset, num_epochs=10, lr=1e-2,
          batch_size=100, optimizer=None, num_mc_steps=20,
          mc_lr=1., verbose=True,
          ckpt_callbacks: List[CheckpointCallback]=None):
    if ckpt_callbacks is None:
        ckpt_callbacks = []
    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True
    )
    # Determine the device type:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(f"training on {len(dataset)} samples using {device}.")
    global_step = 0
    model_samples = None
    for epoch in range(num_epochs):
        losses = []
        for i_batch, sample_batched in enumerate(dataloader):
            global_step += 1
            data_sample = sample_batched[0].to(device)

            # Get model samples, either from replay buffer or noise.
            if model_samples is None:
                model_samples = deque(
                    [net.prior_scale * torch.randn_like(data_sample)])
            elif len(model_samples) > MAX_REPLAY:
                model_samples.popleft()
            replay_sample = random.choice(model_samples)
            noise_sample = net.prior_scale * torch.randn_like(replay_sample)
            mask = torch.rand(replay_sample.shape[0]) < REPLAY_PROB
            while len(mask.shape) < len(replay_sample.shape):
                # Add extra feature-dims
                mask.unsqueeze_(dim=-1)
            model_sample = torch.where(mask.to(device), replay_sample,
                                       noise_sample)

            net.eval()
            for mc_step in range(num_mc_steps):
                model_sample = net.sample_fantasy(model_sample,
                                                  lr=mc_lr).detach()
            model_samples.append(model_sample)

            # Forward gradient:
            net.train()
            net.zero_grad()
            loss = (net(data_sample) - net(model_sample)).mean()
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_value_(net.parameters(), 1e2)
            optimizer.step()
            losses.append(loss)
            if verbose:
                print(f"on epoch {epoch}, batch {i_batch}, loss: {loss}")
            tb_writer.add_scalar(tag="loss/objective", scalar_value=loss, global_step=global_step)
            for callback in ckpt_callbacks:
                callback(net=net, data_sample=data_sample,
                         model_sample=model_sample, epoch=epoch,
                         global_step=global_step, validation=False)

        for callback in ckpt_callbacks:
            callback(net=net, data_sample=data_sample,
                     model_sample=model_sample, epoch=epoch,
                     global_step=global_step, validation=True)

        print(f"on epoch {epoch}, loss: {sum(losses) / len(losses)}")

    return net, optimizer
