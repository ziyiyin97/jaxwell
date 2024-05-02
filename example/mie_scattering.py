# %%
# ruff: noqa: E402
# %%
# This is needed to enable JAX's double-precision mode, see
# https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#Double-(64bit)-precision
# for additional info.
import os

import jaxwell.utils

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax

jax.config.update("jax_enable_x64", True)


# import jax
import jax.numpy as np
import jaxwell
import jaxwell.operators
import matplotlib.pyplot as plt
import numpy as onp
import treams as tr

# Check to make sure double-precision is enabled.
assert jaxwell.utils.double_precision_enabled()


# %%
def plot_field(field, mask=False, abs=True, vmax=0.1, name="out"):
    if mask:
        field = [onp.where(eps_sphere[0] < 3, s, onp.nan) for s in field]
    fig, axs = plt.subplots(1, 4, figsize=(6.3, 2), width_ratios=[2, 2, 2, 0.3])
    for i, ax in enumerate(axs[:3]):
        f = field[i][:, :, num_pixels // 2]

        im = ax.pcolormesh(
            positions[0][:, :, 0],
            positions[1][:, :, 0],
            onp.abs(f) if abs else onp.imag(f),
            vmin=0 if abs else -vmax,
            vmax=vmax,
            cmap="viridis" if abs else "RdBu",
        )
        ax.axis("off")

    fig.colorbar(im, cax=axs[-1])
    plt.savefig(f"{name}.png")


# %%
# Build the structure and source


def center_px(shape):
    """center in units of px"""
    return onp.array([s / 2 - 0.5 for s in shape])


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

    center = onp.array(center).reshape((-1,) + (1,) * 3)
    arr = onp.linalg.norm(np.indices(shape) - center, axis=0)
    return arr <= radius


# %%
eps_bg = 1
eps_fg = 12
dx = 40
wlen = dx * 30

k0 = omega = 2 * onp.pi / wlen
basis = tr.SphericalWaveBasis.default(3)
# print(basis)
num_pixels = 50  # 130
R = 5  # Radius of the sphere
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


x = y = z = onp.arange(num_pixels) * dx
cent = center_fl(shape, dx)
cent[0] = cent[0] + 1  # to add asymmetry
print(cent)

positions = onp.meshgrid(x, y, z, indexing="ij")
positions_spherical = to_spherical_coordinates(*positions, k0, cent)

params = jaxwell.Params(
    pml_ths=((10, 10), (10, 10), (10, 10)),
    pml_omega=omega,
    eps=1e-6,
    max_iters=int(1e6),
)

eps_sphere = []
for component in range(3):
    shift = np.ones(3)
    shift = shift.at[component].set(0.5)

    shifted_sphere = onp.array(structure(R, shape, center=cent / dx + shift))
    eps_sphere.append(shifted_sphere * (eps_fg - eps_bg) + eps_bg)
mode = basis[1]
# field_inc = tr.special.vsw_rA(mode[1], mode[2], *positions_spherical, mode[3])
field_inc_sph = tr.special.vsw_rN(mode[1], mode[2], *positions_spherical)


pos_sph = onp.moveaxis(onp.array(positions_spherical), 0, -1)
field_inc = onp.empty([num_pixels] * 3 + [3], dtype=complex)
for ix in range(num_pixels):
    for iy in range(num_pixels):
        for iz in range(num_pixels):
            vec = field_inc_sph[ix, iy, iz]
            pos = pos_sph[ix, iy, iz]
            field_inc[ix, iy, iz, :] = tr.special.vsph2car(vec, pos)

field_inc_sph = onp.moveaxis(field_inc_sph, -1, 0)
field_inc = onp.moveaxis(field_inc, -1, 0)
# %%
plot_field(field_inc_sph, vmax=0.3, name="inc_sph", abs=True)
plot_field(field_inc, vmax=0.3, name="inc", abs=False)
# %%
image = np.array(eps_sphere)[:, :, :, num_pixels // 2] / 9
image = onp.moveaxis(image, 0, -1)
plt.imshow(image)
# %%
b = (
    onp.array([-(omega**2) * (eps_sphere[0] - eps_bg)] * 3) * field_inc
)  # make sure to do propper decomposition later
b = tuple(b)

z = tuple([omega**2 * eps for eps in eps_sphere])
# %%
field_scat, err, curl = jaxwell.fdfd.solve_impl(z, b, params=params, incl_curl=True)

# %%
plot_field(field_scat, vmax=0.02, name="Escat")


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
plot_field(curl2, vmax=0.005, name="curl2")
# %%
plot_field(curl, vmax=0.005, name="curl")
# %%
