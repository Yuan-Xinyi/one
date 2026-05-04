"""Chinese-character drawing pipeline on top of the trained DiT q0 prior.

Frames each character as a list of polylines (one per pen-down stroke), parameterized
by canonical [-1, 1]² coordinates. A scene-placement step (size, position, orientation)
maps canonical → world frame on the desk plane. Each stroke is then tokenized in the
training format and fed to DiT + IK refine + tracker (the same pipeline that
``visualize_q0_rollout`` uses for a single composite task).

The DiT is used as a **stroke-feasibility prior**: for each candidate stroke, the model
samples N q0 candidates; "feasible" = at least one rolls out cleanly through the tracker.
"""
