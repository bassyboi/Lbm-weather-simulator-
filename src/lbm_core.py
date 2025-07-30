"""Minimal LBM core implementation using a D3Q19 lattice."""
from __future__ import annotations

import numpy as np

# Lattice velocities for D3Q19
C = np.array([
    [0, 0, 0],
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
    [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
    [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
], dtype=np.int8)

W = np.array([
    1/3,
    1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36,
], dtype=float)


class LBMCore:
    """Very small-scale LBM demonstration solver."""

    def __init__(self, shape: tuple[int, int, int], tau: float = 0.6):
        self.shape = shape
        self.tau = tau
        self.omega = 1.0 / tau
        self.f = np.zeros((19, *shape), dtype=float)
        self.rho = np.ones(shape, dtype=float)
        self.u = np.zeros((3, *shape), dtype=float)
        # initialise distributions to equilibrium
        for i in range(19):
            self.f[i, ...] = W[i]

    def collide_and_stream(self):
        # compute macroscopic fields
        self.rho = self.f.sum(axis=0)
        EPSILON = 1e-10  # Small constant to prevent division by zero
        self.u = np.tensordot(C.T, self.f, axes=(1,0)) / (self.rho + EPSILON)

        # equilibrium distribution
        cu = np.einsum('ia,axy->ixy', C, self.u)
        feq = np.empty_like(self.f)
        for i in range(19):
            feq[i] = self.rho * W[i] * (1 + 3*cu[i] + 4.5*cu[i]**2 - 1.5*(self.u**2).sum(axis=0))
        self.f += self.omega * (feq - self.f)

        # streaming step with periodic boundaries
        new_f = np.zeros_like(self.f)
        for i, c in enumerate(C):
            new_f[i] = np.roll(self.f[i], shift=c, axis=(0,1,2))
        self.f = new_f

    def run(self, steps: int = 1):
        for _ in range(steps):
            self.collide_and_stream()

