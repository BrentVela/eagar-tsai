"""Cellular-automaton model of columnar dendritic solidification.

Solutal CA in the style of Zhu & Stefanescu (2001/2007) with
Rappaz & Gandin style capture:

- Explicit finite-difference solute diffusion in the liquid, flux-weighted by
  local liquid fraction (no-flux into solid).
- Interface cells advance by relaxing the local liquid concentration toward
  the equilibrium value C_l* = C0 + (T - T_L + Gamma*kappa) / m
  (Gibbs-Thomson), i.e.  dfs = A(theta) * (C_l* - C_l) / (C_l* (1 - k)).
- Curvature kappa from the counting-cell method on a 5x5 neighborhood.
- 4-fold interface-normal anisotropy A(theta) = 1 + eps*cos(4(theta - theta0)).
- Solute partition k at the interface; rejected solute is pushed into
  neighboring liquid on capture (microsegregation is tracked in cs).
- Frozen-temperature approximation: T(z, t) = T_L(C0) + G * (z - z0 - V t),
  i.e. isotherms sweep upward at the pulling velocity V.

Two modes share the same engine:

- 'directional': seeded bottom layer under an imposed G and V. Primary arm
  spacing (PDAS, lambda_1) emerges from Mullins-Sekerka breakup of the planar
  front followed by growth competition between trunks; secondary arm spacing
  (SDAS, lambda_2) from noise-triggered sidebranching and coarsening.
  With G/V above the constitutional-undercooling limit (G/V >= DT0/D_l, see
  planar_params) the same mode instead sustains a stable planar front.
- 'equiaxed': a single seed growing freely in a uniformly undercooled melt
  (G = 0) with continued cooling; the 4-fold anisotropy selects four primary
  arms and sidebranches develop behind the tips.

Default parameters approximate Al-3wt%Cu.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def add_repo_deps() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    for dirname in [".video_deps", ".deps"]:
        deps = repo_root / dirname
        if deps.exists() and str(deps) not in sys.path:
            sys.path.insert(0, str(deps))


LIQUID, INTERFACE, SOLID = np.uint8(0), np.uint8(1), np.uint8(2)


@dataclass
class CAParams:
    # Domain (cells); growth is upward (row 0 = top of the melt pool / hot end)
    ny: int = 360
    nx: int = 560
    dx: float = 1.5e-6            # cell size [m]

    # Al-3wt%Cu-like alloy
    c0: float = 3.0               # nominal composition [wt%]
    k_part: float = 0.14          # partition coefficient
    m_liq: float = -2.6           # liquidus slope [K/wt%]
    diff_l: float = 3.0e-9        # liquid diffusivity [m^2/s]
    gibbs_thomson: float = 2.4e-7 # Gibbs-Thomson coefficient [K m]

    # Process conditions (directional solidification; D/V ~ 20 um boundary layer)
    mode: str = "directional"     # 'directional' or 'equiaxed'
    grad_t: float = 3.0e4         # thermal gradient G [K/m]
    v_pull: float = 1.5e-4        # isotherm velocity V [m/s]
    undercooling0: float = 4.0    # equiaxed: initial melt undercooling [K]
    cool_rate: float = 10.0       # equiaxed: continued cooling rate [K/s]

    # CA numerics
    aniso_eps: float = 0.60       # 4-fold kinetic anisotropy strength
    theta0: float = 0.0           # grain orientation [rad] (0 => <10> along G)
    growth_gain: float = 0.7      # relaxation factor on dfs per step
    dfs_max: float = 0.05         # interface speed cap [cells/step] (~3x V here)
    capture_fs: float = 0.99      # solid fraction at which a cell fully captures
    noise_amp: float = 0.25       # multiplicative growth noise (sidebranch trigger)
    fo_step: float = 0.20         # diffusion Fourier number D*dt/dx^2 per step
    seed: int = 7

    steps: int = 32000
    stop_row: int = 12            # directional: stop when solid reaches this row
    init_steady_front: bool = False  # start from the steady-state solute boundary layer
    tail_steps: int = 7000        # keep evolving the mush after tips reach the top
    tail_noise_amp: float | None = None  # growth noise during the tail (None = unchanged)
    tail_cool_boost: float = 1.0  # multiply equiaxed cooling rate during the tail
    frame_every: int = 200

    @property
    def dt(self) -> float:
        return self.fo_step * self.dx ** 2 / self.diff_l


@dataclass
class CAResult:
    params: CAParams
    fs: np.ndarray                # final solid fraction
    cs: np.ndarray                # solid composition (microsegregation)
    cl: np.ndarray                # liquid composition
    state: np.ndarray
    frames: list[np.ndarray] = field(default_factory=list)        # fs snapshots
    frame_cs: list[np.ndarray] = field(default_factory=list)      # cs snapshots


def _box_sum(arr: np.ndarray, radius: int) -> np.ndarray:
    """Sum over a (2r+1)^2 neighborhood via separable shifts (edge-padded)."""
    pad = np.pad(arr, radius, mode="edge")
    n = 2 * radius + 1
    horiz = np.zeros_like(pad)
    for i in range(n):
        horiz[:, radius:-radius] += pad[:, i : i + arr.shape[1]]
    out = np.zeros_like(arr)
    for i in range(n):
        out += horiz[i : i + arr.shape[0], radius:-radius]
    return out


def _shift4(arr: np.ndarray) -> list[np.ndarray]:
    """North/south/west/east neighbors with edge padding."""
    p = np.pad(arr, 1, mode="edge")
    return [p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]]


def simulate(params: CAParams | None = None) -> CAResult:
    p = params or CAParams()
    ny, nx, dx = p.ny, p.nx, p.dx
    rng = np.random.default_rng(p.seed)

    fs = np.zeros((ny, nx), dtype=np.float32)
    cl = np.full((ny, nx), p.c0, dtype=np.float32)
    cs_acc = np.zeros((ny, nx), dtype=np.float32)  # integral of k*Cl*dfs
    state = np.full((ny, nx), LIQUID, dtype=np.uint8)

    rows = np.arange(ny)[:, None]
    if p.mode == "equiaxed":
        # single seed in the middle of a uniformly undercooled melt
        yy, xx = np.mgrid[0:ny, 0:nx]
        seeded = (yy - ny // 2) ** 2 + (xx - nx // 2) ** 2 <= 4
        z = np.zeros((ny, nx), dtype=np.float32)
        z_liq0 = 0.0
    else:
        # solid layer at the bottom with a slightly perturbed front so the
        # Mullins-Sekerka instability has something to amplify
        base = 4 + (1.5 + 1.5 * np.sin(2 * np.pi * np.arange(nx) / 70.0)
                    + rng.normal(0.0, 0.8, nx)).round().astype(int)
        seeded = rows >= (ny - base[None, :])
        # height above the bottom [m] and initial liquidus position
        z = ((ny - 1) - rows) * dx * np.ones((1, nx), dtype=np.float32)
        z_liq0 = float(base.mean() + 2) * dx
    fs[seeded] = 1.0
    state[seeded] = SOLID
    cs_acc[seeded] = p.k_part * p.c0

    if p.mode != "equiaxed" and p.init_steady_front:
        # steady growth from t = 0: interface at the solidus isotherm with the
        # steady exponential solute boundary layer (Cl* = c0/k, decay D/V) so
        # the front tracks the isotherms immediately (skips the initial
        # transient, which would otherwise destabilize even a stable front)
        dt0 = -p.m_liq * p.c0 * (1.0 - p.k_part) / p.k_part  # freezing range
        z_front = float(base.mean()) * dx
        z_liq0 = z_front + dt0 / p.grad_t
        decay = np.exp(-np.maximum(z - z_front, 0.0) * p.v_pull / p.diff_l)
        cl = (p.c0 + (p.c0 / p.k_part - p.c0) * decay).astype(np.float32)
        cs_acc[seeded] = p.c0  # steady-state solid composition is c0

    # von Neumann capture keeps diagonal grid filling from swamping the
    # physical 4-fold anisotropy
    solid_nb = sum(_shift4((state == SOLID).astype(np.float32)))
    state[(solid_nb > 0) & (state == LIQUID)] = INTERFACE

    frames: list[np.ndarray] = []
    frame_cs: list[np.ndarray] = []
    tip_arrival: int | None = None

    extra_cool = 0.0
    for step in range(p.steps):
        t = step * p.dt
        in_tail = tip_arrival is not None
        if p.mode == "equiaxed":
            if in_tail:
                extra_cool += (p.tail_cool_boost - 1.0) * p.cool_rate * p.dt
            temp_minus_tl = -(p.undercooling0 + p.cool_rate * t + extra_cool)  # uniform, G = 0
        else:
            temp_minus_tl = p.grad_t * (z - z_liq0 - p.v_pull * t)  # T - T_L(C0)

        # --- solute diffusion among non-solid cells (fluxes masked at solid,
        # edge padding gives no-flux boundaries); explicit and stable for
        # fo_step <= 0.25
        phi = np.where(state != SOLID, 1.0 - fs, 0.0).astype(np.float32)
        open_cell = (state != SOLID).astype(np.float32)
        cp = np.pad(cl, 1, mode="edge")
        op = np.pad(open_cell, 1, mode="edge")
        cc, oc = cp[1:-1, 1:-1], op[1:-1, 1:-1]
        net = np.zeros_like(cl)
        for csh, osh in (
            (cp[:-2, 1:-1], op[:-2, 1:-1]),
            (cp[2:, 1:-1], op[2:, 1:-1]),
            (cp[1:-1, :-2], op[1:-1, :-2]),
            (cp[1:-1, 2:], op[1:-1, 2:]),
        ):
            net += osh * (csh - cc)
        cl += p.fo_step * net * oc
        if p.mode != "equiaxed":
            cl[0, :] = p.c0  # far-field melt at the hot end

        iface = state == INTERFACE
        if not iface.any():
            break

        # --- curvature (counting-cell, 5x5) and interface normal angle
        occupied = np.where(state == SOLID, 1.0, fs).astype(np.float32)
        kappa = (1.0 - 2.0 * _box_sum(occupied, 2) / 25.0) / dx
        smooth = _box_sum(occupied, 1) / 9.0
        gy, gx = np.gradient(smooth)
        theta = np.arctan2(gy, gx)
        aniso = 1.0 + p.aniso_eps * np.cos(4.0 * (theta - p.theta0))

        # --- local equilibrium liquid concentration (Gibbs-Thomson)
        c_eq = p.c0 + (temp_minus_tl + p.gibbs_thomson * kappa) / p.m_liq
        c_eq = np.maximum(c_eq, 1e-3)

        noise_amp = p.noise_amp
        if in_tail and p.tail_noise_amp is not None:
            noise_amp = p.tail_noise_amp
        dfs = p.growth_gain * aniso * (c_eq - cl) / (c_eq * (1.0 - p.k_part))
        dfs *= 1.0 + noise_amp * (rng.random((ny, nx), dtype=np.float32) - 0.5)
        dfs = np.clip(dfs, 0.0, p.dfs_max)
        dfs[~iface] = 0.0

        # --- capture decision: a cell that would (nearly) exhaust its liquid
        # this step solidifies completely; otherwise it partitions smoothly.
        old_liq = np.maximum(1.0 - fs, 0.0)
        captured = iface & ((fs + dfs >= p.capture_fs) | (dfs >= 0.85 * old_liq))
        growing = iface & ~captured & (dfs > 0)

        # smooth growth: solid takes k*Cl, remaining liquid is enriched
        new_liq = np.maximum(old_liq - dfs, 1e-3)
        cl_grown = cl * (old_liq - p.k_part * dfs) / new_liq
        cs_acc[growing] += (p.k_part * cl * dfs)[growing]
        fs[growing] += dfs[growing]
        cl[growing] = cl_grown[growing]

        # capture: whole remaining liquid solidifies at k*Cl; the rejected
        # (1-k)*Cl*old_liq solute mass is pushed into neighboring liquid
        if captured.any():
            cs_acc[captured] += (p.k_part * cl * old_liq)[captured]
            excess = np.where(captured, (1.0 - p.k_part) * cl * old_liq, 0.0).astype(np.float32)
            liq_w = np.where((state != SOLID) & ~captured, np.maximum(phi, 0.05), 0.0)
            nb_w = sum(_shift4(liq_w))
            share = np.where(nb_w > 0, excess / np.maximum(nb_w, 1e-6), 0.0)
            recv = np.zeros_like(cl)
            for sh in _shift4(share):
                recv += sh
            add = recv * liq_w  # each neighbor takes weight-proportional mass
            cl += add / np.maximum(phi, 0.05)
            fs[captured] = 1.0
            state[captured] = SOLID
            cl[captured] = 0.0

            solid_nb = sum(_shift4((state == SOLID).astype(np.float32)))
            state[(solid_nb > 0) & (state == LIQUID)] = INTERFACE

        if step % p.frame_every == 0 or step == p.steps - 1:
            frames.append(fs.astype(np.float16))
            frame_cs.append(
                np.where(fs > 0, cs_acc / np.maximum(fs, 1e-3), 0.0).astype(np.float16))

        # once tips approach the domain edge, let secondary arms develop a
        # while longer before stopping
        if tip_arrival is None:
            if p.mode == "equiaxed":
                near_edge = ((fs[:14] >= 0.5).any() or (fs[-14:] >= 0.5).any()
                             or (fs[:, :14] >= 0.5).any() or (fs[:, -14:] >= 0.5).any())
            else:
                near_edge = (fs[:p.stop_row] >= 0.5).any()
            if near_edge:
                tip_arrival = step
        if tip_arrival is not None and step >= tip_arrival + p.tail_steps:
            frames.append(fs.astype(np.float16))
            frame_cs.append(
                np.where(fs > 0, cs_acc / np.maximum(fs, 1e-3), 0.0).astype(np.float16))
            break

    cs = np.where(fs > 0, cs_acc / np.maximum(fs, 1e-3), 0.0)
    return CAResult(params=p, fs=fs, cs=cs, cl=cl, state=state,
                    frames=frames, frame_cs=frame_cs)


# ----------------------------------------------------------------------------
# PDAS / SDAS measurement
# ----------------------------------------------------------------------------

def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of True runs."""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask_1d.view(np.int8), [0]))))
    return [(int(s), int(e - s)) for s, e in zip(idx[::2], idx[1::2])]


def measure_pdas(fs: np.ndarray, dx: float, row_frac: float = 0.55) -> tuple[float, list[float]]:
    """lambda_1 from trunk centers crossing a horizontal line behind the tips."""
    ny, nx = fs.shape
    solid_rows = np.flatnonzero((fs >= 0.5).any(axis=1))
    top = solid_rows.min()
    row = int(top + row_frac * ((ny - 1) - top))
    band = fs[max(row - 3, 0): row + 4] >= 0.5
    line = band.mean(axis=0) > 0.5
    centers = [s + w / 2.0 for s, w in _runs(line) if w >= 2]
    if len(centers) < 2:
        return float("nan"), centers
    spacings = np.diff(centers) * dx
    return float(np.median(spacings)), centers


def measure_sdas(fs: np.ndarray, dx: float, trunk_centers: list[float],
                 offset_cells: int = 5) -> float:
    """lambda_2 from solid-arm crossings along vertical lines beside trunks."""
    ny, nx = fs.shape
    solid = fs >= 0.5
    spacings: list[float] = []
    for c in trunk_centers:
        for sign in (-1, 1):
            col = int(round(c)) + sign * offset_cells
            if not 0 <= col < nx:
                continue
            profile = solid[:, col]
            rows = np.flatnonzero(profile)
            if rows.size == 0:
                continue
            # only the sidebranched region: above the coherent bottom mush
            segs = _runs(profile)
            arm_centers = [s + w / 2.0 for s, w in segs if 1 <= w <= 25]
            if len(arm_centers) >= 3:
                d = np.diff(arm_centers) * dx
                spacings.extend(d[(d > 2 * dx) & (d < 40e-6)].tolist())
    return float(np.median(spacings)) if spacings else float("nan")


# ----------------------------------------------------------------------------
# Rendering (matches the deck's ca_modes styling)
# ----------------------------------------------------------------------------

PANEL_W, PANEL_H = 900, 560


def render_frame(fs: np.ndarray, cs: np.ndarray, c0: float, k: float,
                 mush: bool = False):
    from PIL import Image, ImageDraw, ImageFilter

    fs = fs.astype(np.float32)
    cs = cs.astype(np.float32)
    solid = fs >= 0.5
    h, w = solid.shape
    y, x = np.mgrid[0:h, 0:w]
    thermal = (0.65 * (1.0 - y / h) + 0.35 * (x / w))[..., None]

    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:] = np.array([232, 238, 240], dtype=np.float32) * (0.84 + 0.16 * thermal)

    # dendrite color modulated by microsegregation: solute-lean cores lighter,
    # enriched late-solidifying regions darker
    base = np.array([65, 126, 181], dtype=np.float32)
    seg = np.clip((cs - k * c0) / (c0 - k * c0 + 1e-6), 0.0, 1.6)[..., None]
    shade = 0.98 - 0.30 * seg + 0.10 * thermal
    dendrite = base * shade + 14
    if mush:
        # partially solidified cells blend toward the dendrite color by fs
        mushy = (fs >= 0.12) & ~solid
        wgt = np.clip(fs / 0.5, 0.0, 1.0)[..., None] * 0.85
        img[mushy] = (img * (1.0 - wgt) + dendrite * wgt)[mushy]
    img[solid] = dendrite[solid]

    pad = np.pad(solid, 1, mode="constant")
    nb = sum(pad[a: a + h, b: b + w].astype(np.int8)
             for a in range(3) for b in range(3)) - solid.astype(np.int8)
    edge = solid & (nb < 8)
    img[edge] = img[edge] * 0.72 + np.array([24, 28, 32], dtype=np.float32) * 0.28

    im = Image.fromarray(np.uint8(np.clip(img, 0, 255))).resize(
        (PANEL_W, PANEL_H), Image.Resampling.BICUBIC)
    im = im.filter(ImageFilter.SMOOTH)
    draw = ImageDraw.Draw(im, "RGBA")
    draw.rectangle([0, 0, PANEL_W - 1, PANEL_H - 1], outline=(33, 37, 41, 170), width=3)
    return im


def add_arrow(im):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(im, "RGBA")
    x = PANEL_W - 82
    draw.line([x, PANEL_H - 62, x, PANEL_H - 210], fill=(255, 255, 255, 230), width=8)
    draw.polygon([(x, PANEL_H - 210), (x - 23, PANEL_H - 166), (x + 23, PANEL_H - 166)],
                 fill=(255, 255, 255, 230))
    return im


def render_annotated(result: CAResult, path: Path) -> tuple[float, float]:
    """Final frame with measured lambda_1 / lambda_2 annotations."""
    from PIL import ImageDraw, ImageFont

    p = result.params
    lam1, centers = measure_pdas(result.fs, p.dx)
    lam2 = measure_sdas(result.fs, p.dx, centers)

    im = render_frame(result.fs, result.cs, p.c0, p.k_part)
    draw = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except OSError:
        font = ImageFont.load_default()

    sx = PANEL_W / p.nx
    if len(centers) >= 2 and np.isfinite(lam1):
        i = len(centers) // 2
        x0, x1 = centers[i - 1] * sx, centers[i] * sx
        yy = 60
        draw.line([x0, yy, x1, yy], fill=(20, 24, 28, 235), width=5)
        for xx in (x0, x1):
            draw.line([xx, yy - 14, xx, yy + 14], fill=(20, 24, 28, 235), width=5)
        draw.text(((x0 + x1) / 2, yy + 20),
                  f"λ₁ = {lam1 * 1e6:.0f} µm",
                  fill=(20, 24, 28, 255), font=font, anchor="ma")
    if np.isfinite(lam2):
        draw.text((24, PANEL_H - 58),
                  f"λ₂ ≈ {lam2 * 1e6:.0f} µm (secondary arms)",
                  fill=(20, 24, 28, 255), font=font)
    im.save(path)
    return lam1, lam2


def render_outputs(result: CAResult, out_dir: Path, *,
                   basename: str = "ca_columnar_dendritic",
                   arrow: bool = True, annotate: bool = True,
                   mush: bool = False, fps: int = 24) -> None:
    add_repo_deps()
    import imageio.v2 as imageio

    out_dir.mkdir(parents=True, exist_ok=True)
    p = result.params

    def styled(fs_snap: np.ndarray, cs: np.ndarray):
        im = render_frame(fs_snap, cs, p.c0, p.k_part, mush=mush)
        return add_arrow(im) if arrow else im

    final = styled(result.fs, result.cs)
    final.save(out_dir / f"{basename}.png")

    with imageio.get_writer(out_dir / f"{basename}.mp4", fps=fps,
                            codec="libx264", quality=8, format="FFMPEG") as writer:
        for fs_snap, cs in zip(result.frames, result.frame_cs):
            writer.append_data(np.array(styled(fs_snap, cs)))
        for _ in range(int(1.5 * fps)):
            writer.append_data(np.array(final))

    if annotate:
        lam1, lam2 = render_annotated(result, out_dir / "ca_pdas_sdas_annotated.png")
        print(f"G = {p.grad_t:.2e} K/m, V = {p.v_pull:.2e} m/s, "
              f"cooling rate = {p.grad_t * p.v_pull:.0f} K/s")
        print(f"measured PDAS lambda_1 = {lam1 * 1e6:.1f} um")
        print(f"measured SDAS lambda_2 = {lam2 * 1e6:.1f} um")


def planar_params() -> CAParams:
    """Stable planar growth: dilute alloy + steep gradient + slow pulling.

    Constitutional undercooling limit: V_crit = G*D_l/DT0 with
    DT0 = |m|*c0*(1-k)/k ~ 4.8 K here, so V_crit ~ 1.9e-4 m/s; V = 4e-5 m/s
    keeps the front planar and initial perturbations decay.
    """
    return CAParams(c0=0.3, grad_t=3.0e5, v_pull=4.0e-5, noise_amp=0.05,
                    init_steady_front=True, capture_fs=0.7, steps=55000,
                    stop_row=185, tail_steps=300, frame_every=275, seed=5)


def equiaxed_params() -> CAParams:
    return CAParams(mode="equiaxed", steps=60000, undercooling0=3.0,
                    cool_rate=20.0, gibbs_thomson=1.2e-6, tail_steps=2500,
                    tail_noise_amp=0.05, tail_cool_boost=2.5,
                    frame_every=100, seed=11)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "figures" / "ca_modes"
    render_outputs(simulate(), out_dir)
    render_outputs(simulate(equiaxed_params()), out_dir,
                   basename="ca_dendritic", arrow=False, annotate=False,
                   mush=True)
    render_outputs(simulate(planar_params()), out_dir,
                   basename="ca_planar", annotate=False)


if __name__ == "__main__":
    main()
