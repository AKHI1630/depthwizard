"""Generate a fake nadir satellite image with building-like structures at 512×512."""
from PIL import Image
import numpy as np

np.random.seed(42)
arr = np.full((512, 512, 3), 80, dtype=np.uint8)

# Ground texture
noise = np.random.randint(-10, 10, arr.shape, dtype=np.int16)
arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Buildings: bright rectangles (rooftops)
buildings = [
    (50, 50, 100, 90),
    (150, 30, 210, 80),
    (250, 120, 310, 200),
    (80, 250, 140, 330),
    (200, 280, 290, 350),
    (350, 50, 420, 120),
    (380, 200, 470, 300),
    (100, 400, 180, 470),
    (300, 380, 400, 460),
    (420, 380, 490, 480),
]
for x0, y0, x1, y1 in buildings:
    brightness = np.random.randint(180, 240)
    arr[y0:y1, x0:x1] = brightness
    arr[y0:y1, x0:x1] += np.random.randint(-5, 5, (y1-y0, x1-x0, 3)).astype(np.uint8)

Image.fromarray(arr).save("test_satellite.png")
print("Saved test_satellite.png (512×512)")
