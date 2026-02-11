# Copyright 2021 Google LLC
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

"""Pseudospectral equations."""

import dataclasses
from typing import Callable, Optional

import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import forcings
from jax_cfd.base import grids
from jax_cfd.spectral import forcings as spectral_forcings
from jax_cfd.spectral import time_stepping
from jax_cfd.spectral import types
from jax_cfd.spectral import utils as spectral_utils


TimeDependentForcingFn = Callable[[float], types.Array]
RandomSeed = int
ForcingModule = Callable[[grids.Grid, RandomSeed], TimeDependentForcingFn]


@dataclasses.dataclass
class KuramotoSivashinsky(time_stepping.ImplicitExplicitODE):
  """Kuramoto–Sivashinsky (KS) equation split in implicit and explicit parts.

  The KS equation is
    u_t = - u_xx - u_xxxx - 1/2 * (u ** 2)_x

  Implicit parts are the linear terms and explicit parts are the non-linear
  terms.

  Attributes:
    grid: underlying grid of the process
    smooth: smooth the non-linear term using the 3/2-rule
  """
  grid: grids.Grid
  smooth: bool = True

  def __post_init__(self):
    self.kx, = self.grid.rfft_axes()
    self.two_pi_i_k = 2j * jnp.pi * self.kx
    self.linear_term = -self.two_pi_i_k ** 2 - self.two_pi_i_k ** 4
    self.rfft = spectral_utils.truncated_rfft if self.smooth else jnp.fft.rfft
    self.irfft = spectral_utils.padded_irfft if self.smooth else jnp.fft.irfft

  def explicit_terms(self, uhat):
    """Non-linear parts of the equation, namely `- 1/2 * (u ** 2)_x`."""
    uhat_squared = self.rfft(jnp.square(self.irfft(uhat)))
    return -0.5 * self.two_pi_i_k * uhat_squared

  def implicit_terms(self, uhat):
    """Linear parts of the equation, namely `- u_xx - u_xxxx`."""
    return self.linear_term * uhat

  def implicit_solve(self, uhat, time_step):
    """Solves for `implicit_terms`, implicitly."""
    # TODO(dresdner) the same for all linear terms. generalize/refactor?
    return 1 / (1 - time_step * self.linear_term) * uhat


@dataclasses.dataclass
class ForcedBurgersEquation(time_stepping.ImplicitExplicitODE):
  """Burgers' Equation with the option to add a time-dependent forcing function."""
  viscosity: float
  grid: grids.Grid
  seed: int = 0
  forcing_module: Optional[
      ForcingModule] = spectral_forcings.random_forcing_module
  _forcing_fn = None

  def __post_init__(self):
    self.kx, = self.grid.rfft_axes()
    self.two_pi_i_k = 2j * jnp.pi * self.kx
    self.linear_term = self.viscosity * self.two_pi_i_k ** 2
    self.rfft = spectral_utils.truncated_rfft
    self.irfft = spectral_utils.padded_irfft
    if self.forcing_module is None:
      self._forcing_fn = lambda t: jnp.zeros(1)
    else:
      self._forcing_fn = self.forcing_module(self.grid, self.seed)

  def explicit_terms(self, state):
    uhat, t = state
    dudx = self.two_pi_i_k * uhat

    f = self._forcing_fn(t)
    fhat = jnp.fft.rfft(f)

    advection = - self.rfft(self.irfft(uhat) * self.irfft(dudx))

    return (fhat + advection, 1.0)

  def implicit_terms(self, state):
    uhat, _ = state
    return (self.linear_term * uhat, 0.0)

  def implicit_solve(self, state, time_step):
    uhat, t = state
    return (1 / (1 - time_step * self.linear_term) * uhat, t)


def BurgersEquation(viscosity: float, grid: grids.Grid, seed: int = 0):
  """Standard, unforced Burgers' equation."""
  return ForcedBurgersEquation(
      viscosity=viscosity, grid=grid, seed=seed, forcing_module=None)


# pylint: disable=invalid-name
def _get_grid_variable(arr,
                       grid,
                       bc=boundaries.periodic_boundary_conditions(2),
                       offset=(0.5, 0.5)):
  return grids.GridVariable(grids.GridArray(arr, offset, grid), bc)


@dataclasses.dataclass
class NavierStokes2D(time_stepping.ImplicitExplicitODE):
  """Breaks the Navier-Stokes equation into implicit and explicit parts.

  Implicit parts are the linear terms and explicit parts are the non-linear
  terms.

  Attributes:
    viscosity: strength of the diffusion term
    grid: underlying grid of the process
    smooth: smooth the advection term using the 2/3-rule.
    forcing_fn: forcing function, if None then no forcing is used.
    drag: strength of the drag. Set to zero for no drag.
  """
  viscosity: float
  grid: grids.Grid
  drag: float = 0.
  operator_order: float = 1.0
  smooth: bool = True
  forcing_fn: Optional[Callable[[grids.Grid], types.Spectral2DForcingFn_vort]] = None
  velocity_bckgrd: Optional[Callable[[grids.Grid], types.VelocityFn]] = None
  k_filter: float = None
  _forcing_fn_with_grid = None
  _velocity_fn_with_grid = None

  def __post_init__(self):
    self.kx, self.ky = self.grid.rfft_mesh()
    #self.laplace = (jnp.pi * 2j)**2 * (self.kx**2 + self.ky**2)
    self.n_laplace =  - ( (jnp.pi * 2j)**2 * ( - self.kx**2 - self.ky**2 ) )**self.operator_order
    if self.k_filter is None:
      self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
    else:
      self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid,self.k_filter)
    self.linear_term = self.viscosity * self.n_laplace - self.drag

    # setup the forcing function with the caller-specified grid.
    if self.forcing_fn is not None:
      self._forcing_fn_with_grid = self.forcing_fn(self.grid)
    if self.velocity_bckgrd is not None:
      self._velocity_fn_with_grid = self.velocity_bckgrd(self.grid)
    self.velocity_solve = spectral_utils.vorticity_to_velocity(self.grid)

  def explicit_terms(self, vorticity_hat):
    #velocity_solve = spectral_utils.vorticity_to_velocity(self.grid)
    vxhat, vyhat = self.velocity_solve(vorticity_hat)
    vx, vy = jnp.fft.irfftn(vxhat), jnp.fft.irfftn(vyhat)

    grad_x_hat = 2j * jnp.pi * self.kx * vorticity_hat
    grad_y_hat = 2j * jnp.pi * self.ky * vorticity_hat
    grad_x, grad_y = jnp.fft.irfftn(grad_x_hat), jnp.fft.irfftn(grad_y_hat)

    advection = -(grad_x * vx + grad_y * vy)
    

    if self.velocity_bckgrd is not None:
      u_bg, v_bg = self._velocity_fn_with_grid()
      advection += -(grad_x * u_bg.data + grad_y * v_bg.data)

    advection_hat = jnp.fft.rfftn(advection)
    if self.smooth is not None:
      advection_hat *= self.filter_

    terms = advection_hat

    if self.forcing_fn is not None:
      # Forcing should provide curl of forcing.
      curl_f_hat = self._forcing_fn_with_grid()
      terms += curl_f_hat

    return terms

  def implicit_terms(self, vorticity_hat):
    return self.linear_term * vorticity_hat

  def implicit_solve(self, vorticity_hat, time_step):
    return 1 / (1 - time_step * self.linear_term) * vorticity_hat


# pylint: disable=g-doc-args,g-doc-return-or-yield,invalid-name
def ForcedNavierStokes2D(viscosity, grid, smooth,
                          dissipation_order: int = 1):
  """Sets up the flow that is used in Kochkov et al. [1].

  The authors of [1] based their work on Boffetta et al. [2].

  References:
    [1] Machine learning–accelerated computational fluid dynamics. Dmitrii
    Kochkov, Jamie A. Smith, Ayya Alieva, Qing Wang, Michael P. Brenner, Stephan
    Hoyer Proceedings of the National Academy of Sciences May 2021, 118 (21)
    e2101784118; DOI: 10.1073/pnas.2101784118.
    https://doi.org/10.1073/pnas.2101784118

    [2] Boffetta, Guido, and Robert E. Ecke. "Two-dimensional turbulence."
    Annual review of fluid mechanics 44 (2012): 427-451.
    https://doi.org/10.1146/annurev-fluid-120710-101240
  """
  wave_number = 4
  offsets = ((0, 0), (0, 0))
  # pylint: disable=g-long-lambda
  forcing_fn = lambda grid: forcings.kolmogorov_forcing(
      grid, k=wave_number, offsets=offsets)
  return NavierStokes2D(
      viscosity,
      grid,
      drag=0.1,
      operator_order=dissipation_order,
      smooth=smooth,
      forcing_fn=forcing_fn)


@dataclasses.dataclass
class NonlinearSchrodinger(time_stepping.ImplicitExplicitODE):
  """Nonlinear schrodinger equation split in implicit and explicit parts.

  The NLS equation is
    `psi_t = -i psi_xx/8 - i|psi|^2 psi/2`

  Attributes:
    grid: underlying grid of the process
    smooth: smooth the non-linear by upsampling 2x in fourier and truncating
  """
  grid: grids.Grid
  smooth: bool = True

  def __post_init__(self):
    self.kx, = self.grid.fft_axes()
    assert len(self.kx) % 2 == 0, "Odd grid sizes not supported, try N even"
    self.two_pi_i_k = 2j * jnp.pi * self.kx
    self.fft = spectral_utils.truncated_fft_2x if self.smooth else jnp.fft.fft
    self.ifft = spectral_utils.padded_ifft_2x if self.smooth else jnp.fft.ifft

  def explicit_terms(self, psihat):
    """Non-linear part of the equation `-i|psi|^2 psi/2`."""
    psi = self.ifft(psihat)
    ipsi_cubed = 1j * psi * jnp.abs(psi)**2
    ipsi_cubed_hat = self.fft(ipsi_cubed)
    return -ipsi_cubed_hat / 2

  def implicit_terms(self, psihat):
    """The diffusion term `-i psi_xx/8` to be handled implicitly."""
    return -1j * psihat * self.two_pi_i_k**2 / 8

  def implicit_solve(self, psihat, time_step):
    """Solves for `implicit_terms`, implicitly."""
    return psihat / (1 - time_step * (-1j * self.two_pi_i_k**2 / 8))

@dataclasses.dataclass
class CompressibleNavierStokes2D(time_stepping.ImplicitExplicitODE):
    viscosity: float
    beta: float
    gamma: float
    kappa: float
    grid: grids.Grid
    alpha: float = 0
    chi: float = 0
    smooth: bool = True
    forcing_fn: Optional[Callable[[grids.Grid], types.Spectral2DForcingFn]] = None
    _forcing_fn = None

    def __post_init__(self):
        # Build spectral mesh
        self.kx, self.ky = self.grid.rfft_mesh()
        self.ikx = 2j * jnp.pi * self.kx
        self.iky = 2j * jnp.pi * self.ky
        self.Nx, self.Ny = self.grid.shape
        # Wavenumber magnitude squared
        self.k2 = (2 * jnp.pi)**2 * (self.kx**2 + self.ky**2) + 1e-14
 
        # Fourier-space linear operator coefficients (no dt here)
        self.A   = -(self.alpha + self.viscosity * self.k2)            # u-u coupling
        self.CB  =  self.k2 * (self.kappa + self.chi * self.k2)        # for φ-denominator
        self.Bx0 = - self.ikx * (self.kappa + self.chi * self.k2)      # φ→u_x
        self.By0 = - self.iky * (self.kappa + self.chi * self.k2)      # φ→u_y
 
        # Spectral filter for smoothing explicit terms
        filt = spectral_utils.brick_wall_filter_2d(self.grid)
        self.filter_vec = filt[None, :, :]
        if self.forcing_fn is not None:
          self._forcing_fn_with_grid = self.forcing_fn(self.grid)
 
    def explicit_terms(self, state_hat):
        """
        Compute N(x^n): nonlinear advective terms for (u_x, u_y) and zero for φ.
        """
        ux_hat, uy_hat, phi_hat = state_hat
 
        # Transform velocities back to real space
        u = jnp.fft.irfft2(jnp.stack([ux_hat, uy_hat], axis=0), s=(self.Nx, self.Ny))
 
        # Compute gradients in real space
        grad_ux = jnp.fft.irfft2(jnp.stack([self.ikx * ux_hat, self.iky * ux_hat], axis=0),
                                  s=(self.Nx, self.Ny))
        grad_uy = jnp.fft.irfft2(jnp.stack([self.ikx * uy_hat, self.iky * uy_hat], axis=0),
                                  s=(self.Nx, self.Ny))
 
        # Advective + dilatation contributions
        adv = jnp.stack([
            -self.beta * (u[0] * grad_ux[0] + u[1] * grad_ux[1])
            - self.gamma * u[0] * (grad_ux[0] + grad_uy[1]),
            -self.beta * (u[0] * grad_uy[0] + u[1] * grad_uy[1])
            - self.gamma * u[1] * (grad_ux[0] + grad_uy[1])
        ], axis=0)
 
        # Return spectral advective terms
        adv_hat = jnp.fft.rfft2(adv, s=(self.Nx, self.Ny))
        if self.smooth:
            adv_hat *= self.filter_vec
 
        if self.forcing_fn is not None:
          # Forcing should provide curl of forcing.
          fu, fv = self._forcing_fn_with_grid()
          #terms += curl_f_hat
          adv_hat += jnp.stack([fu,fv],axis = 0)


        return adv_hat[0], adv_hat[1], jnp.zeros_like(phi_hat)
 
    def implicit_terms(self, state_hat):
        ux, uy, phi = state_hat
        Lu_x = self.A * ux + self.Bx0 * phi
        Lu_y = self.A * uy + self.By0 * phi
        Lphi = - (self.ikx * ux + self.iky * uy)
        return Lu_x, Lu_y, Lphi

    def implicit_solve(self, y_hat, dt):
        yx, yy, yphi = y_hat
 
        # Precompute
        den_u     = 1 - dt * self.A
        div_y     = self.ikx * yx + self.iky * yy
        denom_phi = den_u + dt**2 * self.CB
 
        # 1) φ-solve by Schur complement
        num_phi = den_u * yphi - dt * div_y
        phi_n1  = num_phi / denom_phi
 
        # 2) velocity back-substitution
        ux_n1 = (yx + dt * self.Bx0 * phi_n1) / den_u
        uy_n1 = (yy + dt * self.By0 * phi_n1) / den_u
 
        return ux_n1, uy_n1, phi_n1
    

@dataclasses.dataclass
class NavierStokes3D(time_stepping.ImplicitExplicitODE):
  """Breaks the Navier-Stokes equation into implicit and explicit parts.

  Implicit parts are the linear terms and explicit parts are the non-linear
  terms.

  Attributes:
    viscosity: strength of the diffusion term
    grid: underlying grid of the process
    smooth: smooth the advection term using the 2/3-rule.
    forcing_fn: forcing function, if None then no forcing is used.
    drag: strength of the drag. Set to zero for no drag.
  """
  viscosity: float
  grid: grids.Grid
  drag: float = 0.
  operator_order: float = 1.0
  smooth: bool = True
  k_filter: float = None
  forcing_fn: Optional[Callable[[grids.Grid], types.Spectral2DForcingFn]] = None
  _forcing_fn_with_grid = None
  fu = None

  def __post_init__(self):
    kx,ky,kz = self.grid.rfft_mesh()
    self.Dx, self.Dy, self.Dz = jnp.pi * 2j * kx, jnp.pi * 2j * ky, jnp.pi * 2j * kz
    self.n_laplace =  - ( - self.Dx**2 - self.Dy**2 - self.Dz**2 )**self.operator_order
    nabla2 = self.Dx**2 + self.Dy**2 + self.Dz**2
    self.ID2 = jnp.where(nabla2 == 0, 1.0, 1/nabla2)
    if self.k_filter is None:
      self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
    else:
      self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid,self.k_filter)
    self.linear_term = self.viscosity * self.n_laplace - self.drag

    if self.forcing_fn is not None:
            self._forcing_fn_with_grid = self.forcing_fn(self.grid)
            self.fu = self._forcing_fn_with_grid()
            
  def explicit_terms(self, state):

    """ Velocity field is of the form (Nx,Ny,Nz,i)
        where i is the vector index """
    U_hat = state
    U_space = jnp.fft.irfftn( U_hat, axes = (0,1,2) )

    udx_U = self.Dx[...,None] * jnp.fft.rfftn( U_space[:,:,:,0:1] * U_space, axes = (0,1,2) )
    vdy_U = self.Dy[...,None] * jnp.fft.rfftn( U_space[:,:,:,1:2] * U_space, axes = (0,1,2) )
    wdz_U = self.Dz[...,None] * jnp.fft.rfftn( U_space[:,:,:,2:3] * U_space, axes = (0,1,2) )

    adv = udx_U + vdy_U + wdz_U

    div_adv = self.Dx * adv[...,0] + self.Dy * adv[...,1] + self.Dz * adv[...,2]
    div_adv_scaled = div_adv * self.ID2
    proj = jnp.stack([div_adv_scaled * self.Dx,
                      div_adv_scaled * self.Dy,
                      div_adv_scaled * self.Dz],axis=-1)
    adv -= proj
    
    terms =  - adv

    if self.fu is not None:
      terms += self.fu

    if self.smooth is not None:
      terms *= self.filter_[...,None]

    return terms

  def implicit_terms(self, state):
    U_hat = state
    return self.linear_term[...,None] * U_hat

  def implicit_solve(self, state, time_step):
    U_hat = state
    return 1 / (1 - time_step * self.linear_term[...,None]) * U_hat
  

@dataclasses.dataclass
class MHD(time_stepping.ImplicitExplicitODE):
  """Breaks the 2D MHD equations into implicit and explicit parts.

    Implicit parts are linear (diffusion) and explicit parts are nonlinear (advection, Lorentz force).

    Attributes:
        viscosity: strength of vorticity diffusion
        resistivity: strength of magnetic diffusion
        grid: underlying computational grid
        smooth: smooth the advection term using the 2/3-rule
        forcing_fn: optional forcing function
        drag: drag term applied to velocity
        velocity_bckgrd: optional background velocity field
        magnetic_bckgrd: optional background magnetic field
  """
  viscosity: float
  resistivity: float
  grid: grids.Grid
  drag: float = 0.
  operator_order: float = 1.0
  smooth: bool = True
  forcing_fn: Optional[Callable[[grids.Grid], types.Spectral2DForcingFn_vort]] = None
  velocity_bckgrd: Optional[Callable[[grids.Grid], types.VelocityFn]] = None
  magnetic_bckgrd: Optional[Callable[[grids.Grid], types.VelocityFn]] = None
  k_filter: Optional[float] = None

  _forcing_fn_with_grid = None
  _velocity_fn_with_grid = None
  _magnetic_fn_with_grid = None

  def __post_init__(self):
    self.kx, self.ky = self.grid.rfft_mesh()
    self.n_laplace = -((2j * jnp.pi) ** 2 * (-self.kx**2 - self.ky**2)) ** self.operator_order
    # spectral filter
    if self.k_filter is None:
        self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
    else:
        self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid, self.k_filter)
    # linear terms for velocity (viscosity) and magnetic field (resistivity)
    self.linear_vorticity = self.viscosity * self.n_laplace - self.drag
    self.linear_magnetic = self.resistivity * self.n_laplace
    # setup forcing / background functions
    if self.forcing_fn is not None:
        self._forcing_fn_with_grid = self.forcing_fn(self.grid)
    if self.velocity_bckgrd is not None:
        self._velocity_fn_with_grid = self.velocity_bckgrd(self.grid)
    if self.magnetic_bckgrd is not None:
        self._magnetic_fn_with_grid = self.magnetic_bckgrd(self.grid)
    # velocity solve from vorticity
    self.velocity_solve = spectral_utils.vorticity_to_velocity(self.grid)

  def explicit_terms(self, state):
    """Compute the explicit (nonlinear) part of 2D MHD: advection + Lorentz force"""
    w_hat, j_hat = state  # vorticity and current

    # velocity from vorticity
    vx_hat, vy_hat = self.velocity_solve(w_hat)
    vx, vy = jnp.fft.irfftn(vx_hat), jnp.fft.irfftn(vy_hat)

    # magnetic field from current
    bx_hat, by_hat = self.magnetic_solve(j_hat)
    bx, by = jnp.fft.irfftn(bx_hat), jnp.fft.irfftn(by_hat)

    # gradients
    grad_wx_hat = 2j * jnp.pi * self.kx * w_hat
    grad_wy_hat = 2j * jnp.pi * self.ky * w_hat
    grad_wx = jnp.fft.irfftn(grad_wx_hat)
    grad_wy = jnp.fft.irfftn(grad_wy_hat)

    grad_jx_hat = 2j * jnp.pi * self.kx * j_hat
    grad_jy_hat = 2j * jnp.pi * self.ky * j_hat
    grad_jx = jnp.fft.irfftn(grad_jx_hat)
    grad_jy = jnp.fft.irfftn(grad_jy_hat)

    # advection
    w_adv = -(grad_wx * vx + grad_wy * vy)
    j_adv = -(grad_jx * vx + grad_jy * vy)  # B advected by velocity

    # Lorentz term: curl(B x J) simplified in vorticity formulation
    lorentz = grad_jx * by - grad_jy * bx
    w_adv += lorentz

    # background velocity / magnetic field
    if self.velocity_bckgrd is not None:
        u_bg, v_bg = self._velocity_fn_with_grid()
        w_adv += -(grad_wx * u_bg.data + grad_wy * v_bg.data)
        j_adv += -(grad_jx * u_bg.data + grad_jy * v_bg.data)
    if self.magnetic_bckgrd is not None:
        bx_bg, by_bg = self._magnetic_fn_with_grid()
        w_adv += grad_jx * by_bg.data - grad_jy * bx_bg.data
        j_adv += 0  # optional: usually only affects vorticity

    # Fourier transform and filtering
    w_adv_hat = jnp.fft.rfftn(w_adv)
    j_adv_hat = jnp.fft.rfftn(j_adv)

    if self.smooth:
        w_adv_hat *= self.filter_
        j_adv_hat *= self.filter_

    # forcing
    if self.forcing_fn is not None:
        w_adv_hat += self._forcing_fn_with_grid()  # curl forcing

    return (w_adv_hat, j_adv_hat)

  def implicit_terms(self, state):
      """Linear implicit terms (diffusion, drag)"""
      w_hat, j_hat = state
      return (self.linear_vorticity * w_hat, self.linear_magnetic * j_hat)

  def implicit_solve(self, state, dt):
      """Solve (I - dt * L) w_hat = rhs for implicit linear part"""
      w_hat, j_hat = state
      w_new = w_hat / (1 - dt * self.linear_vorticity)
      j_new = j_hat / (1 - dt * self.linear_magnetic)
      return (w_new, j_new)