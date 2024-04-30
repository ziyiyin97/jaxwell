# %%
# ruff: noqa: E402
# %%
# This is needed to enable JAX's double-precision mode, see
# https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#Double-(64bit)-precision
# for additional info.
from jax.config import config

import jaxwell.operators

config.update("jax_enable_x64", True)
# import jax
import jax.numpy as np
import jaxwell
import matplotlib.pyplot as plt
import numpy as onp
import treams as tr

# import treams

# Check to make sure double-precision is enabled.
assert np.zeros((1,), np.float64).dtype == np.float64


# %%
def plot_field(field, mask=True, vmax=0.1):
    if mask:
        field = [onp.where(eps_sphere[0] < 3, s, onp.nan) for s in field]
    fig, axs = plt.subplots(1, 4, figsize=(6.3, 2), width_ratios=[2, 2, 2, 0.3])
    for i, ax in enumerate(axs[:3]):
        im = ax.pcolormesh(
            positions[0][:, :, 0],
            positions[1][:, :, 0],
            onp.abs(field[i][:, :, num_pixels // 2]),
            vmin=0,
            vmax=vmax,
        )
        ax.axis("off")

    fig.colorbar(im, cax=axs[-1])


# %%
# Build the structure and source


def center_px(shape):
    """center in units of px"""
    return np.array([s / 2 - 0.5 for s in shape])


def center_fl(shape, dx) -> float:
    """center in nm (float)"""
    return dx * center_px(shape)


def structure(radius, shape, center: tuple[int, int, int] | None = None):
    """Builds a ball of radius `radius`

    For simplicity, we do not take into account the offsets between the x-, y-,
    and z-components of epsilon in the Yee cell.

    Args:
      radius: radius of the spehere in pixels.
      shape: `(xx, yy, zz)` tuple defining the shape of the simulation.
      center: specifies the center of the sphere as a tuple.
        If `None` the sphere is centered in the simulation region
    """
    if center is None:
        center = center_px(shape)

    center = np.array(center).reshape((-1,) + (1,) * 3)
    arr = np.linalg.norm(np.indices(shape) - center, axis=0)
    return 0.5 * (
        (arr <= radius - 1) * 0.25 + (arr <= radius) * 0.5 + (arr <= radius + 1) * 0.25
    )


# %%
eps_bg = 2
eps_fg = 12
dx = 40
wlen = dx * 15

k0 = omega = 2 * onp.pi / wlen
basis = tr.SphericalWaveBasis.default(3)
num_pixels = 140
R = 20  # Radius of the sphere
print(f"Radius: {R}")
shape = (num_pixels,) * 3


# %%
def to_spherical_coordinates(x, y, z, k0, center):
    """Converts to spherical coordinates in units of k0"""
    x, y, z = (comp - cent for comp, cent in zip([x, y, z], center))
    r = onp.sqrt(x**2 + y**2 + z**2)
    theta = onp.arccos(z / r)
    phi = onp.sign(y) * onp.arccos(x / onp.sqrt(x**2 + y**2))
    theta[r == 0] = 0
    phi[r == 0] = 0
    return r * k0, theta, phi


x = y = z = np.arange(num_pixels) * dx
cent = center_fl(shape, dx)
cent = cent.at[0].set(cent[0] + 0.1)  # to add asymmetry
print(cent)

positions = onp.meshgrid(x, y, z, indexing="ij")
positions_spherical = to_spherical_coordinates(*positions, k0 / dx, cent)

params = jaxwell.Params(
    pml_ths=((10, 10), (10, 10), (10, 10)),
    pml_omega=omega,
    eps=1e-6,
    max_iters=int(1e6),
)

sphere = onp.array(structure(R, shape, center=cent / dx))
eps_sphere = [
    sphere * (eps_fg - eps_bg) + eps_bg
] * 3  # not super accurate (do subpixel smoothing)
mode = basis[0]
field_inc = tr.special.vsw_rA(mode[1], mode[2], *positions_spherical, mode[3])
field_inc = tr.special.vsw_rN(mode[1], mode[2], *positions_spherical)
field_inc = onp.moveaxis(field_inc, -1, 0)
# %%
plot_field(field_inc)
# %%
plot_field(eps_sphere, mask=False, vmax=None)
# %%
b = (
    onp.array([-(omega**2) * (eps_sphere[0] - eps_bg)] * 3) * field_inc
)  # make sure to do propper decomposition later
b = tuple(b)

z = tuple([omega**2 * eps for eps in eps_sphere])
# %%
field_scat, err, curl = jaxwell.fdfd.solve_impl(z, b, params=params, incl_curl=True)

# %%
plot_field(field_scat, vmax=0.05)


# %%
def curlE_dual(E):
    diff_fn = jaxwell.operators.spatial_diff
    y = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        y.append(diff_fn(E[k], axis=j) - diff_fn(E[j], axis=k))
    return jaxwell.vecfield.VecField(*y)


E = jaxwell.vecfield.from_tuple(field_scat)
curl2 = jaxwell.vecfield.to_tuple(curlE_dual(E))
plot_field(curl2, vmax=0.02)
# %%
plot_field(curl, vmax=0.02)
# %%
