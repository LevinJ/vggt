import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime
from visual_util import segment_sky, download_file_from_url


def segment_sky_from_image(image_path: str, model_path: str = "/home/levin/workspace/nerf/forward/vggt/skyseg.onnx") -> np.ndarray:
    """
    Perform sky segmentation on a single image.

    Args:
        image_path (str): Path to the input RGB image.
        model_path (str): Path to the ONNX model for sky segmentation.

    Returns:
        np.ndarray: Sky mask for the input image.
    """
    # Ensure the ONNX model exists
    if not os.path.exists(model_path):
        print(f"Downloading {model_path}...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", model_path)

    # Load the ONNX model
    skyseg_session = onnxruntime.InferenceSession(model_path)

    # Perform sky segmentation
    sky_mask = segment_sky(image_path, skyseg_session)

    return sky_mask==0


if __name__ == "__main__":
    # Test the function with the provided image
    test_image_path = "/media/levin/DATA/nerf/new_es8/stereo/20250701/left_images/1751336593.4671859741.png"

    print(f"Segmenting sky for image: {test_image_path}")
    sky_mask = segment_sky_from_image(test_image_path)

    # Load the original image
    original_image = cv2.imread(test_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Create an image with sky pixels set to black
    sky_removed_image = original_image.copy()
    sky_removed_image[sky_mask] = [0, 0, 0]  # Assuming sky mask is binary with 255 for sky pixels

    # Display the original image, sky mask, and sky-removed image side by side
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Sky Mask")
    plt.imshow(sky_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Sky Removed Image")
    plt.imshow(sky_removed_image)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
