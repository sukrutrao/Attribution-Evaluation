import torch
import torchvision
import numpy as np
import os
from torchvision import transforms as transforms
from tqdm import tqdm
import PIL
import argparse


def main(args):
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    labels = np.genfromtxt(args.labels_path, delimiter=" ", dtype=str)
    num_images = labels.shape[0]

    out_imgs = torch.empty((num_images, 3, 224, 224))
    out_labels = torch.empty((num_images, 1), dtype=torch.long)
    for img_idx in tqdm(range(len(labels))):
        img_name = labels[img_idx][0]
        img_label = int(labels[img_idx][1])
        img = PIL.Image.open(os.path.join(args.images_path, img_name)).convert("RGB")
        transformed_img = transform(img)
        out_imgs[img_idx] = transformed_img
        out_labels[img_idx] = img_label

    out_dict = {'data': out_imgs, 'labels': out_labels.to(
        torch.long), 'input_dims': (3, 224, 224), 'num_classes': 1000, 'scale': 1}
    os.makedirs(args.save_path, exist_ok=False)
    torch.save(out_dict, os.path.join(args.save_path, "test.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts the ImageNet validation set to a PyTorch tensor with the standard transformations, for use by the script to generate the grid dataset.")
    parser.add_argument('--images_path', type=str, required=True,
                        help="Path to directory containing ImageNet images.")
    parser.add_argument('--labels_path', type=str, required=True,
                        help="Path to text file containing labels for the dataset, with each line containing the image name, followed by a space, and followed by the label ID.")
    parser.add_argument('--save_path', type=str, required=True,
                        help="Path to save the converted dataset.")

    args = parser.parse_args()
    main(args)
