"""M1 — CV + Pose prototype pipeline.

Classical-CV approximation of the M1 module from the research plan:
frame grab -> pop-plate homography -> die/pip keypoints -> PnP 6DoF pose,
plus a HITL annotation overlay and a clip collector.
"""

__version__ = "0.1.0"