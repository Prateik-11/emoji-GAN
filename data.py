import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(img_folder_path, img_size):
    """
    This time, we load the dataset with the following 
    transformations applied:
    """
    data_transforms = [
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.RandomHorizontalFlip(), # Remove this line if horizontally flipping doesn't make sense with your images
        torchvision.transforms.ToTensor(), # Scales data to [0,1] 
        torchvision.transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    return torchvision.datasets.ImageFolder(root=img_folder_path, 
                                            transform=torchvision.transforms.Compose(
                                                                                data_transforms))

def get_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

def show_tensor_image(image):
    reverse_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda t: (t + 1) / 2),
        torchvision.transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        torchvision.transforms.Lambda(lambda t: t * 255.),
        torchvision.transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        torchvision.transforms.ToPILImage(),
    ])
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

def show_images(img_folder_path, num_samples=32, cols=8):
    """ 
    Plots some samples from the dataset 
    """
    dataset = torchvision.datasets.ImageFolder(img_folder_path)
    fig = plt.figure(figsize=(15,9))
    fig.patch.set_alpha(0) 
    axs = fig.axes
    for ax in axs:
        ax.remove()
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        ax = plt.subplot(int(num_samples/cols + 1), cols, i + 1)
        ax.axis("off")
        plt.imshow(img[0])


# data = load_transformed_dataset()
# dataloader = torch.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)