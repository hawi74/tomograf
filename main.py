import math
from os import path, listdir

import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_float
from skimage.io import imread
from skimage.transform import radon, rescale, iradon
from skimage.measure import compare_mse


IMAGES_PATH = 'tomograf-zdjecia'

no_of_detectors = int(input("Number of detectors (default: 100): ") or 100)
scan_range = float(input("Scan range (default: 180): ") or 180)
no_of_scans = int(input("Number of scans (default: 360): ") or 360)
filtering = bool((input("Filtering: [Y/n]: ") or 'Y') == 'Y')
available_images = listdir(IMAGES_PATH)
images = '\n'.join(
    f'[{n}] {image_path}'
    for n, image_path
    in enumerate(available_images)
)
selected_image_index = int(
    input(f'Select image: (default: 0)\nf{images}\n') or 0,
)
image_name = available_images[selected_image_index]

print(f"""=====================
Using {no_of_detectors} detectors
Scan range is set to {scan_range}°
Doing {no_of_scans} scans
Filtering is {'on' if filtering else 'off'}
Image source: {image_name}
=====================""")

orig_image = imread(path.join(IMAGES_PATH, image_name), as_grey=True)

fig, (orig_ax, sinogram_ax, restored_ax) = plt.subplots(3, 1, figsize=(8, 4.5))

orig_ax.set_title(image_name)
orig_ax.imshow(orig_image, cmap=plt.cm.Greys_r)

sinogram_ax.set_title("Sinogram")

theta = np.linspace(0., scan_range, no_of_scans, endpoint=False)
sinogram = radon(orig_image, theta=theta, circle=True)
sinogram_ax.set_title("Radon transform\n(Sinogram)")
sinogram_ax.set_xlabel("Projection angle (deg)")
sinogram_ax.set_ylabel("Projection position (pixels)")
sinogram_ax.imshow(
    sinogram,
    cmap=plt.cm.Greys_r,
    extent=(0, 180, 0, sinogram.shape[0]),
    aspect='auto',
)

restored_ax.set_title("Restored")
restored_image = iradon(
    sinogram,
    theta=theta,
    circle=True,
    filter="ramp" if filtering else None,
    output_size=no_of_detectors,
)
restored_ax.imshow(rescale(restored_image, 100 / restored_image.shape[0]), cmap=plt.cm.Greys_r)

scale = restored_image.shape[0] / orig_image.shape[0]

rmse = round(
    math.sqrt(
        compare_mse(
            img_as_float(rescale(orig_image, scale=scale)),
            img_as_float(restored_image),
        ),
    ),
    8,
)

restored_ax.text(
    200,
    50,
    f'RMSE({image_name}, restored) = \n{rmse}',
)

fig.tight_layout()
plt.show()