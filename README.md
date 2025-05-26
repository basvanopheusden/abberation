# Abberation

This repository contains a simple matplotlib animation that demonstrates two types of optical aberration.

The animation models parallel rays incident on a curved optical interface.  At
each animation step the curvature smoothly transitions from a spherical surface
to a nearly flat plane.  Refraction at the interface is computed with Snell's
law, so rays that strike the surface further from the optical axis bend more and
focus at a different location than rays near the center.  As the surface flattens
the rays eventually fail to converge, demonstrating how spherical aberration can
arise from the shape of the refracting element.

For additional background on the physics see the
[Wikipedia article on optical aberration](https://en.wikipedia.org/wiki/Optical_aberration).

Run the animation with:

```bash
python3 aberration_animation.py
```

Several parameters can be overridden from the command line.  The most
useful options are:

| Option | Meaning |
| ------ | ------- |
| `--n-rays` | Number of rays traced through the optic (default `10`). |
| `--frames` | Total number of animation frames (default `120`). |
| `--max-in-angle` | Final incoming angle in radians at the edges of the beam (default `0.3`). |
| `--plane-x` | X position of the optical interface when it becomes flat (default `0.5`). |
| `--radius` | Radius of the spherical surface at the start of the animation (default `0.5`). |
| `--far-radius` | Effective radius when the surface is nearly a plane (default `50`). |
| `--aperture` | Half height of the optical element and view window (default `0.6`). |
| `--interval` | Delay between animation frames in milliseconds (default `50`). |
| `--x-start` | X coordinate where rays originate (default `-1.0`). |
| `--x-final` | X coordinate where rays terminate (default `1.6`). |
| `--focal-point` | Coordinates of the focal point as `x y` (default `1.2 0.0`). |
| `--xlim` | X-axis limits for the plot (default computed from `x-start` and `x-final`). |

For example, to animate with more rays and a slower frame rate run:

```bash
python3 aberration_animation.py --n-rays 20 --interval 100
```

When run outside the test suite, this script also saves the first and last
frames as `aberration_first.png` and `aberration_last.png`.  In addition the
entire animation is written to `aberration.gif`. Saving is skipped
automatically when `pytest` is running.

A second script `plane_wave_animation.py` demonstrates a different scenario. It begins with a plane wave of horizontal rays striking the curved surface and then gradually flattens the interface while keeping the outgoing rays fixed.
Run it with:

```bash
python3 plane_wave_animation.py
```

Like the main aberration animation script, most parameters can be overridden
from the command line. Useful options include `--n-rays`, `--frames`,
`--plane-x`, `--radius`, `--far-radius`, `--aperture`, `--surf-samples`,
`--interval`, `--n-ratio`, `--x-start` and `--x-final`.

When run outside the test suite, `plane_wave_animation.py` also saves the first
and last frames as `plane_wave_first.png` and `plane_wave_last.png`. The full
animation is stored as `plane_wave.gif`. Saving is skipped automatically during
`pytest` runs.

## Development

Run the tests with:

```bash
pytest -q
```
