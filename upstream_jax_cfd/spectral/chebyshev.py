# Copyright 2025 CNS gatech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Code for chebyshev spectral methods."""

import jax.numpy as jnp
import jax.scipy.fft
from jax import lax
import jax.numpy.fft as jnp_fft

from jax_cfd.base.grids import Grid

import dataclasses
import operator
from jax_cfd.spectral.types import Array
from typing import Any, Callable, Optional, Sequence, Tuple, Union
import jax
import jax_cfd.spectral.utils as utils

@dataclasses.dataclass(init=False, frozen=True)
class cheb_Grid(Grid):
    """ Describes the size and shape for and spectral grid
        Has information of real and spectral grid
    """
    shape: Tuple[int, ...]
    step: Tuple[None, ...]
    domain: Tuple[Tuple[float, float], ...]
    cheb_axes: Tuple[int, ...]

    def __init__(
        self,
        shape: Sequence[int],
        domain: Optional[Union[float, Sequence[Tuple[float, float]]]],
        ):

        shape = tuple(operator.index(s) for s in shape)
        object.__setattr__(self, 'shape', shape)

        if isinstance(domain, (int, float)):
            domain = ((0, domain),) * len(shape)
        else:
            if len(domain) != self.ndim:
                raise ValueError('length of domain does not match ndim: '
                                f'{len(domain)} != {self.ndim}')
            for bounds in domain:
                if len(bounds) != 2:
                    raise ValueError(
                    f'domain is not sequence of pairs of numbers: {domain}')
        domain = tuple((float(lower), float(upper)) for lower, upper in domain)
        
        object.__setattr__(self, 'domain', domain)
        object.__setattr__(self, 'step', (None,) * len(shape))
        object.__setattr__(self, 'cheb_axes', (1,))

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of this grid."""
        return len(self.shape)

    @property
    def cell_faces(self) -> Tuple[Tuple[float, ...]]:
        """Returns the offsets at each of the 'forward' cell faces."""
        raise NotImplementedError('cell_faces() not implemented for cheb_Grid class.')

    def stagger(self, v: Tuple[Array, ...]) -> Any:
        """Places the velocity components of `v` on the `Grid`'s cell faces."""
        raise NotImplementedError('stagger() not implemented for cheb_Grid class.')

    def center(self, v: Any) -> Any:
        """Places all arrays in the pytree `v` at the `Grid`'s cell center."""
        raise NotImplementedError('center() not implemented for cheb_Grid class.')

    def axes(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
        """Returns a tuple of arrays containing the grid points along each axis.

        Returns:
        An tuple of `self.ndim` arrays. The jth return value has shape
        `[self.shape[j]]`.
        """
        coords = []

        for d, ((lower, upper), size) in enumerate(zip(self.domain, self.shape)):
            
            if d in self.cheb_axes:
                if size == 1:
                    coord = jnp.array([0.5 * (lower + upper)])
                else:
                    # Chebyshev-Lobatto points in [-1, 1]
                    i = jnp.arange(size)
                    x_cheb = jnp.cos(jnp.pi * i / (size - 1))
                    # Map to [lower, upper]
                    coord = 0.5 * (lower + upper) + 0.5 * (upper - lower) * x_cheb
            else:
                # Uniform periodic grid: endpoint=False to mimic periodicity
                coord = jnp.linspace(lower, upper, size, endpoint=False)

            coords.append(coord)
    
        return tuple(coords)

    def mesh(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
        """Returns a tuple of arrays containing positions in each grid cell.
        
        For Chebyshev grids, this creates a meshgrid from the non-uniform axes.
        The offset parameter is ignored for Chebyshev dimensions.
        
        Args:
            offset: an optional sequence of length `ndim`. Ignored for Chebyshev grids.
            
        Returns:
            A tuple of `self.ndim` arrays, each of shape `self.shape`.
        """
        axes = self.axes(offset)
        return tuple(jnp.meshgrid(*axes, indexing='ij'))



    def get_chebyshev_transforms(self):
        """Returns the Chebyshev forward and backward transform functions.

        Returns:
            A tuple (forward_transform, backward_transform), where
            - forward_transform: function for Chebyshev forward transform
            - backward_transform: function for Chebyshev backward transform
        """
        return utils.chebyshev_forward_transform, utils.chebyshev_backward_transform
