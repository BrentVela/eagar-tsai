# muMatScale dendrite cases

These cases use ORNL's BSD-licensed [muMatScale](https://github.com/lang-yuan/muMatScale),
not the deck's custom NumPy CA. It was tested with muMatScale commit
`4820401e59ec8959bd8fa9fe03eb645505acc3b4`.

## Equiaxed

Build muMatScale with CMake, then run this case from an empty output directory:

```sh
mpiexec -n 1 /path/to/muMatScale/build/muMatScale \
  -i -c /path/to/eagar-tsai/beamer/mumatscale/equiaxed_dendrite.in
```

The selected image is the high-resolution section at step 8000
(`equiaxed_r0/08000.h5`). At the refined grid this is the closest morphology
match to the selected coarse result, with full primary arms and restrained
secondary branching:

```sh
python3 beamer/mumatscale/render_equiaxed_dendrite.py \
  /path/to/output/equiaxed_r0/08000.h5 \
  beamer/figures/ca_modes/ca_dendritic_mumatscale.png
```

The input is a 440 x 440 x 3 thin slab at 0.5 micrometer in-plane resolution with one
centered nucleus, uniform undercooling, Al-3 wt% Cu-like properties, solutal
diffusion, and Gibbs-Thomson curvature enabled. `FSGrow=0.78`, a
`4.5e-7 m K` Gibbs-Thomson coefficient, and the step-8000 stop time retain full
primary trunks while allowing a small number of secondary arms to mature. The
renderer reconstructs the phase compositions from muMatScale's cell-averaged
`CE` field using `Cl = CE / (1 - (1-k) fs)` and `Cs = k Cl`. Both are mapped
onto separate phase scales. The solid uses the deck's dark-blue family and the
liquid a near-white sky-blue ramp, with stronger blue indicating enrichment.
Bicubic upscaling plus light smoothing suppresses cell stair-stepping; the
finer CA domain supplies the additional spatial detail.

## Columnar

Run the directional-solidification case from another empty output directory:

```sh
mpiexec -n 1 /path/to/muMatScale/build/muMatScale \
  -i -c /path/to/eagar-tsai/beamer/mumatscale/columnar_dendrite.in
```

The slide uses the selected dense-column frame at step 6000
(`columnar_r0/06000.h5`):

```sh
python3 beamer/mumatscale/render_columnar_dendrite.py \
  /path/to/output/columnar_r0/06000.h5 \
  beamer/figures/ca_modes/ca_columnar_mumatscale.png
```

This case is a 600 x 3 x 360 thin slab at 0.5 micrometer resolution, giving a
300 micrometer-wide by 180 micrometer-tall x-z section. A 30 K/mm gradient and
2 mm/s upward isotherm drive growth from twenty closely spaced, slightly
misoriented boundary nuclei. The renderer uses the center plane selected during
the simulation review and makes a square, equal-scale crop with the dendrites
occupying the lower half. Its 900 x 900 output exactly matches the equiaxed
panel and uses the same phase-composition reconstruction, Beamer blue palettes,
bicubic upscaling, and light smoothing.

## Planar

Run the high-gradient planar case from an empty output directory:

```sh
mpiexec -n 1 /path/to/muMatScale/build/muMatScale \
  -i -c /path/to/eagar-tsai/beamer/mumatscale/planar_front.in
```

The slide uses step 8000 (`planar_r0/08000.h5`):

```sh
python3 beamer/mumatscale/render_planar_front.py \
  /path/to/output/planar_r0/08000.h5 \
  beamer/figures/ca_modes/ca_planar_mumatscale.png
```

This 360 x 3 x 360 case uses 0.5 micrometer cells, a 100 K/mm thermal
gradient, and a 2 mm/s upward isotherm. A fully populated, crystal-aligned
bottom nucleation layer merges immediately into a planar envelope. The
renderer takes the x-z center plane and makes an equal-scale 50 x 50-cell crop,
placing the flat front at mid-height in a 900 x 900 image. Its liquid scale is
expanded to 3--7 wt% Cu so far-field liquid remains near white while the narrow
blue band above the dark solid identifies the Cu-enriched boundary layer.
