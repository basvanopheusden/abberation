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
