"""Video preprocessing: frame-rate interpolation and spatial upscaling.

Synthesizes high-frame-rate / high-resolution video from lower-rate input as a
BOOTSTRAP ONLY. Interpolated frames are fabricated and must not be treated as
real high-fps ground truth for physics/calibration.
"""

__version__ = "0.1.0"