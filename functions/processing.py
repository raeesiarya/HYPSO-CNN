import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libraries import *

#######################################################################################
#######################################################################################
#######################################################################################

def load_image(image_path, HEIGHT=598, WIDTH=1092, BANDS=120):
    with open(image_path, 'rb') as f:
        image = cp.fromfile(f, dtype=cp.uint16)
        image = cp.asnumpy(image)
        image = image.reshape((BANDS, HEIGHT, WIDTH))
        image = image.transpose((1, 2, 0))
        image = image.reshape((-1, BANDS))

    image = cut_bands(image)
    
    return image

#######################################################################################

def load_label(label_path, HEIGHT=598, WIDTH=1092):
    label = np.fromfile(label_path, dtype=np.uint8)
    label = label.reshape((HEIGHT, WIDTH))
    label = label - 1
    label = label.flatten()
    return label

#######################################################################################

def global_min_max_normalization(images, verbose=False):
    """
    Normalize the images using global per-band min-max normalization. That is,
    the minimum and maximum values are calculated separately for each spectral band
    across all pixels in the dataset.
    """
    min_values = images.min(dim=0).values
    max_values = images.max(dim=0).values

    if verbose:
        print(colored(f"Checking spectra of the first 10 pixels.", "red"))
        check_normalization(images, "red")
        print("\n")

    new_images = (images - min_values) / (max_values - min_values + 1e-8)

    verify_global_min_max_normalization(images, new_images)
    
    if verbose:
        print("\n")
        print(colored(f"Checking spectra of the first 10 pixels after min-max normalization.", "green"))
        check_normalization(new_images, "green")

    return new_images

#######################################################################################

def verify_global_min_max_normalization(original_images, normalized_images, epsilon=1e-8):
    min_vals = original_images.min(dim=0).values
    max_vals = original_images.max(dim=0).values

    expected = (original_images - min_vals) / (max_vals - min_vals + epsilon)

    if not torch.allclose(normalized_images, expected, atol=1e-6):
        diffs = torch.abs(normalized_images - expected)
        max_diff = diffs.max().item()
        print(colored(f"Normalization mismatch! Maximum deviation: {max_diff}", "red"))
        raise AssertionError("Normalization doesn't follow the formula.")
    else:
        print(colored("All pixels are correctly normalized.","green"))

#######################################################################################

def check_normalization(images, color):
    br1 = 0
    for i in range(images.shape[0]):
        print(colored(images[i], color))
        if br1 == 10:
            break
        br1 += 1
    
#######################################################################################

def cut_bands(image, trim_start=3, trim_end=117):
    return image[:, trim_start:trim_end]

#######################################################################################

class normalization_manager:
    def __init__(self, epsilon=1e-8):
        self.min_vals = None
        self.max_vals = None
        self.epsilon = epsilon

    def fit(self, images):
        self.min_vals = images.min(dim=0).values
        self.max_vals = images.max(dim=0).values
        print(colored("Fitted normalization parameters from training data.", "green"))

    def transform(self, images, verify=False, verbose=False):
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("NormalizationManager not fitted. Call fit(images) first.")
        
        if verbose:
            print(colored(f"Checking spectra of the first 10 pixels.", "red"))
            self.check_normalization(images, "red")
            print("\n")
        
        norm_images = (images - self.min_vals) / (self.max_vals - self.min_vals + self.epsilon)

        if verify:
            self.verify(images, norm_images)

        if verbose:
            print("\n")
            print(colored(f"Checking spectra of the first 10 pixels after min-max normalization.", "green"))
            self.check_normalization(norm_images, "green")
        
        return norm_images
    
    def verify(self, original_images, normalized_images):
        expected = (original_images - self.min_vals) / (self.max_vals - self.min_vals + self.epsilon)

        if not torch.allclose(normalized_images, expected, atol=1e-6):
            diffs = torch.abs(normalized_images - expected)
            max_diff = diffs.max().item()
            print(colored(f"Normalization mismatch! Maximum deviation: {max_diff}", "red"))
            raise AssertionError("Normalization doesn't follow the formula.")
        else:
            print(colored("All pixels are correctly normalized.","green"))

    def check_normalization(self, images, color):
        br1 = 0
        for i in range(images.shape[0]):
            print(colored(images[i], color))
            if br1 == 10:
                break
            br1 += 1