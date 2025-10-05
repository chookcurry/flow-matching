from abc import ABC, abstractmethod

import torch
from torch import Tensor
from tqdm import tqdm

from flow_matching.supervised.odes_sdes import ODE, SDE


class Simulator(ABC):
    @abstractmethod
    def step(self, xt: Tensor, t: Tensor, dt: Tensor, y: Tensor) -> Tensor:
        # xt: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # dt: (B, 1, 1, 1)
        # y: (B, 1, 1, 1)
        # nxt: (B, C, H, W)
        pass

    @torch.no_grad()
    def simulate(self, x: Tensor, ts: Tensor, y: Tensor) -> Tensor:
        # x: (B, C, H, W)
        # ts: (B, num_timesteps, 1, 1, 1)
        # y: (B, 1, 1, 1)
        # x_final: (B, C, H, W)

        num_timesteps = ts.shape[1]
        for i in tqdm(range(num_timesteps - 1)):
            t = ts[:, i]
            dt = ts[:, i + 1] - ts[:, i]
            x = self.step(x, t, dt, y)

        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: Tensor, ts: Tensor, y: Tensor) -> Tensor:
        # x: (B, C, H, W)
        # ts: (B, num_timesteps, 1, 1, 1)
        # y: (B, 1, 1, 1)
        # xs: (B, num_timesteps, C, H, W)

        x_list = [x.clone()]
        num_timesteps = ts.shape[1]
        for t_idx in tqdm(range(num_timesteps - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, y)
            x_list.append(x.clone())

        xs = torch.stack(x_list, dim=1)
        # (B, num_timesteps, C, H, W)

        return xs


class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: Tensor, t: Tensor, dt: Tensor, y: Tensor) -> Tensor:
        return xt + self.ode.drift_coeff(xt, t, y) * dt


class HeunSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: Tensor, t: Tensor, dt: Tensor, y: Tensor) -> Tensor:
        k1 = self.ode.drift_coeff(xt, t, y)
        xt_euler = xt + dt * k1
        k2 = self.ode.drift_coeff(xt_euler, t + dt, y)

        return xt + 0.5 * dt * (k1 + k2)


class RK4Simulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: Tensor, t: Tensor, dt: Tensor, y: Tensor) -> Tensor:
        k1 = self.ode.drift_coeff(xt, t, y)
        k2 = self.ode.drift_coeff(xt + 0.5 * dt * k1, t + 0.5 * dt, y)
        k3 = self.ode.drift_coeff(xt + 0.5 * dt * k2, t + 0.5 * dt, y)
        k4 = self.ode.drift_coeff(xt + dt * k3, t + dt, y)

        return xt + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: Tensor, t: Tensor, dt: Tensor, y: Tensor) -> Tensor:
        return (
            xt
            + self.sde.drift_coeff(xt, t, y) * dt
            + self.sde.diffusion_coeff(xt, t, y) * torch.sqrt(dt) * torch.randn_like(xt)
        )


def record_every(num_timesteps: int, record_every: int) -> Tensor:
    if record_every == 1:
        return torch.arange(num_timesteps)

    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            Tensor([num_timesteps - 1]),
        ]
    )
