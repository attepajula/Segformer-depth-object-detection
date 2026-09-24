"""ADE20K label constants — the classes we surface, and their overlay colours.

Kept framework-free (no mlx / torch import) so ``overlay`` and the backend
loop can use it without pulling in a model runtime.
"""

from __future__ import annotations

ADE20K_NUM_CLASSES = 150

# ADE20K label id -> name, for the structure classes we detect.
ADE20K_TARGETS: dict[int, str] = {
    14: "door",
    38: "railing",
    95: "bannister",
    58: "screen door",
    53: "stairs",
    59: "stairway",
}

# ADE ids that get the depth door-gate.
DOORLIKE = {14, 58}

# fixed overlay colour per class (RGB).
CLASS_COLOR: dict[int, tuple[int, int, int]] = {
    14: (255, 130, 0),    # door        — orange
    38: (60, 220, 60),    # railing     — green
    95: (255, 235, 0),    # bannister   — yellow
    58: (230, 60, 230),   # screen door — magenta
    53: (0, 230, 230),    # stairs      — cyan
    59: (150, 90, 255),   # stairway    — violet
}
CLASS_COLOR_DEFAULT = (0, 180, 255)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
