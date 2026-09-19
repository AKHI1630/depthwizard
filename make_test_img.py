from PIL import Image
import numpy as np

arr = np.zeros((256, 256, 3), dtype=np.uint8)
arr[50:100, 50:100] = 220
arr[150:200, 120:180] = 200
arr[80:130, 160:210] = 180
arr[:, :] = np.clip(arr.astype(int) + np.random.randint(0, 10, arr.shape), 0, 255).astype(np.uint8)
Image.fromarray(arr).save("test_buildings.png")
print("Saved test_buildings.png")
