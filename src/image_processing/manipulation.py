from skimage import io, filters
import numpy as np
from sklearn.decomposition import PCA


class ImageManipulator:
    def __init__(self, image_path: str):
        self.image = self.load_image(image_path)

    def load_image(self, image_path: str):
        image = io.imread(image_path)
        if image.ndim != 5:
            raise ValueError(
                "Image is not 5D (expected dimensions: Z, Time, Channel, X, Y)"
            )
        print("Loaded image shape:", image.shape)
        return image

    def extract_slice(
            self,
            z: int = None,
            time: int = None,
            channel: int = None
    ):
        """
        Extract a slice from a 5D image (Z, Time, Channel, X, Y).
        If an index is None, all data along that dimension is kept.
        """
        data = self.image

        if z is not None:
            if z < 0 or z >= data.shape[0]:
                raise ValueError(
                    f"z value out of range (0 to {data.shape[0]-1})"
                )
            data = data[z]  # Now shape becomes (Time, Channel, X, Y)

        if time is not None:
            if time < 0 or time >= data.shape[0]:
                raise ValueError(
                    f"time value out of range (0 to {data.shape[0]-1})"
                )
            data = data[time]  # Now shape becomes (Channel, X, Y)

        if channel is not None:
            if channel < 0 or channel >= data.shape[0]:
                raise ValueError(
                    f"channel value out of range (0 to {data.shape[0]-1})"
                )
            data = data[channel]  # Now shape becomes (X, Y)

        return data

    def apply_pca(self, n_components: int = 3):
        """
        Flatten the spatial dimensions of the image and perform PCA.
        Assumes image shape: (Z, Time, Channel, X, Y).
        The spatial dimensions (X, Y) are flattened so that each 
        sample is of shape (X*Y,), and PCA is applied.
        The output is reshaped to (Z, Time, Channel, n_components).
        """
        if self.image.ndim != 5:
            raise ValueError("Expected a 5D image for PCA")
        Z, T, C, X, Y = self.image.shape
        # Flatten the spatial dimensions: samples shape = (Z*T*C, X*Y)
        samples = self.image.reshape(Z * T * C, X * Y)
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(samples)  # shape: (Z*T*C, n_components) # noqa
        # Reshape back to (Z, T, C, n_components)
        transformed = transformed.reshape(Z, T, C, n_components)
        return transformed

    def calculate_statistics(self):
        mean = float(np.mean(self.image))
        std_dev = float(np.std(self.image))
        min_val = float(np.min(self.image))
        max_val = float(np.max(self.image))
        return {
            'mean': mean,
            'std_dev': std_dev,
            'min': min_val,
            'max': max_val
        }

    def apply_segmentation(self):
        """
        Apply segmentation (using Otsu thresholding)
        on a representative 2D slice of a 5D image.
        The 5D image is expected to
        have shape (Z, Time, Channel, X, Y).
        This implementation extracts the first slice (z=0, time=0) and then:
          - If there are at least 3 channels,
          a grayscale is computed from the first three.
          - Otherwise, the first channel is used.
        """
        # Extract a representative 2D slice (from z=0 and time=0)
        slice_5d = self.image  # shape: (Z, Time, Channel, X, Y)
        slice_2d_multi = slice_5d[0, 0]  # shape: (Channel, X, Y)

        if slice_2d_multi.shape[0] < 3:
            # Use the first channel directly
            # if fewer than 3 channels are available.
            gray_image = slice_2d_multi[0]
        else:
            # Compute a grayscale image from the first three channels.
            gray_image = (0.2989 * slice_2d_multi[0] +
                          0.5870 * slice_2d_multi[1] +
                          0.1140 * slice_2d_multi[2])

        threshold_value = filters.threshold_otsu(gray_image)
        segmented = (gray_image > threshold_value).astype(np.uint8) * 255
        return segmented
