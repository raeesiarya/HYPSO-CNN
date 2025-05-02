import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libraries import *
from functions.processing import *
from manage_data import read_csv_file

#######################################################################################
#######################################################################################
#######################################################################################

class merged_hyperspectral_dataset(Dataset):
    def __init__(self, list_of_images, list_of_labels=None, normalizer=None):
        """
        Initializes the dataset by loading images and optionally labels, and applies normalization if provided.

        Args:
            list_of_images (list of str): A list of file paths to the image files to be loaded.
            list_of_labels (list of str, optional): A list of file paths to the label files to be loaded. 
                If None, labels will not be loaded. Defaults to None.
            normalizer (object, optional): An object with a `transform` method to normalize the images. 
                If None, no normalization is applied. Defaults to None.

        Attributes:
            images (torch.Tensor): A tensor containing the loaded and optionally normalized images.
            labels (torch.Tensor or None): A tensor containing the loaded labels if provided, otherwise None.
        """

        all_images = []

        if list_of_labels is not None:
            all_labels = []

        for image_path, label_path in tqdm(zip(list_of_images, list_of_labels), 
                                           desc="Loading images and labels", 
                                           total=len(list_of_images)):
            image = load_image(image_path) # (pixels, bands)
            all_images.append(image)

            if list_of_labels is not None:
                label = load_label(label_path) # (pixels, )
                all_labels.append(label)

        self.images = torch.from_numpy(np.concatenate(all_images)).float().contiguous()

        if list_of_labels is not None:
            self.labels = torch.from_numpy(np.concatenate(all_labels)).long()
        else:
            self.labels = None

        if normalizer is not None:
            self.images = normalizer.transform(self.images, verify=True, verbose=False)

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            If `self.labels` is None:
                numpy.ndarray or torch.Tensor: The image at the specified index.
            Otherwise:
                tuple: A tuple containing:
                    - numpy.ndarray or torch.Tensor: The image at the specified index.
                    - numpy.ndarray or torch.Tensor: The corresponding label at the specified index.
        """
        if self.labels is None:
            return self.images[idx]
        else:
            return self.images[idx], self.labels[idx]