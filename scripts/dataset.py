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
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        #print(self.images[idx].shape)
        if self.labels is None:
            return self.images[idx]
        else:
            return self.images[idx], self.labels[idx]

#######################################################################################

"""
bip_paths, dat_paths, png_paths = read_csv_file("csv/train_files.csv")

normalizer = normalization_manager()

raw_dataset = merged_hyperspectral_dataset(bip_paths, dat_paths)
normalizer.fit(raw_dataset.images)
dataset = merged_hyperspectral_dataset(
    bip_paths, dat_paths, normalizer=normalizer
)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Test en enkelt batch
for pixels, labels in dataloader:
    print(colored(f"Pixels shape: {pixels.shape}", "light_blue"))   # Forventet: (128, num_bands)
    print(colored(f"Labels shape: {labels.shape}", "light_green"))  # Forventet: (128,)
    print(colored(f"Unique labels in batch: {torch.unique(labels)}", "light_yellow"))
    break"
"""