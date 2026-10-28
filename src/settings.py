# Design choices made on the validation drives by validate_settings.py (rules fixed in that script; results in
# results/drift/validation.csv and choices.csv). Every experiment script imports them from here.
ETA = 0.1           # quantile-tracking step (beams)
M_SWEEP = 100       # a full sweep happens with probability 1/M
GRID = 6            # GPS-free grid: every 6th beam
HYBRID = (2, 16)    # hybrid: top-2 predicted beams + every 16th beam
