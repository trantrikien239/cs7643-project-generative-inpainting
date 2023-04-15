import numpy as np 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Create a custom tform to prepare images in place205 for inpainting model training
# by randomly select a mask from the masks array and apply it to the image
# and then convert the image to a pytorch tensor

# First, we need to create a custom transform class that applies the mask to the image
class MaskedCocoCrop(object):
    def __init__(self, mask_array):
        self.mask_array = mask_array
        self.n_masks = mask_array.shape[0]

    def __call__(self, sample):
        image = sample
        # Randomly select a mask from the mask array
        mask = self.mask_array[np.random.randint(self.n_masks)]
        # Convert the image to a pytorch tensor
        tensor_image = transforms.ToTensor()(image)
        # Apply the mask to the image
        tensor_masked_image = tensor_image * (1 - mask)
        return {'images': tensor_image, "masked_images": tensor_masked_image}
    
class MaskedDataset(Dataset):
    def __init__(self, ds, mask_array):
        self.ds = ds
        self.mask_array = mask_array
        self.n_masks = mask_array.shape[0]

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        image = self.ds.images[idx].numpy()
        label = self.ds.labels[idx].numpy(fetch_chunks = True).astype(np.int32)
        if image.shape[-1] == 1:
            image = np.stack([image.squeeze()]*3, axis=2)
        mask = self.mask_array[np.random.randint(self.n_masks)]
        sample = {"images": image / 255., 
                  "masked_images": (image * np.expand_dims(1 - mask, axis = 2)) / 255., 
                  "masks": mask,
                  "labels": label}

        return sample