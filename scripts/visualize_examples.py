import torch
import argparse
import os
import numpy as np
from attribution_evaluation.evaluation import visualization
import torchvision
from attribution_evaluation.models import settings


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    test_data_dict = torch.load(os.path.join(args.dataset_path, 'test.pt'))
    scale = test_data_dict["scale"]
    grid_size = scale*scale
    img_dims = test_data_dict["input_dims"][1:]
    if not settings.eval_only_corners(args.setting):
        head_idx = args.head_pos_idx
    else:
        head_list = [0, grid_size-1]
        assert args.head_pos_idx in head_list
        head_idx = head_list.index(args.head_pos_idx)

    imagenet_inv_normalize_transform = torchvision.transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    images = imagenet_inv_normalize_transform(test_data_dict["data"])

    attributions = torch.load(os.path.join(args.attributions_path, 'attributions_' +
                                           args.model + '_' + args.setting + '_' + os.path.basename(args.dataset_path) + '_' + args.exp + '_' + args.config + '_' + str(args.layer) + args.save_suffix + '.pt'))

    fig, _ = visualization.visualize_examples(
        attributions, images, num_examples=args.num_examples, head_idx=head_idx, head_pos_idx=args.head_pos_idx, img_dims=img_dims, scale=scale)
    fig.tight_layout(pad=0, h_pad=-1)
    full_save_path = os.path.join(args.save_path, "examples_" +
                                  args.model + "_" + args.setting + "_" + os.path.basename(args.dataset_path) + '_' + args.exp + '_' + args.config + '_' + str(args.layer) + '_' + str(args.head_pos_idx) + args.save_suffix + ".png")
    print("Saving examples at", full_save_path)
    fig.savefig(full_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualizes examples from each AggAtt bin for a set of attributions for ImageNet data.")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Path of directory containing the dataset")
    parser.add_argument('--seed', type=int, default=1, help="Random seed value")
    parser.add_argument('--model', type=str, required=True,
                        choices=["vgg11", "resnet18"], help="Model to evaluate on")
    parser.add_argument('--setting', type=str, required=True,
                        choices=["GridPG", "DiFull", "DiPart"], help="Setting to evaluate on")
    parser.add_argument('--layer', type=str, required=True, choices=["Input", "Middle", "Final"],
                        help="Layer to evaluate on")
    parser.add_argument('--attributions_path', type=str, required=True,
                        help="Path of directory from which to load attributions")
    parser.add_argument('--save_path', type=str, required=True,
                        help="Path of directory in which to save visualization")
    parser.add_argument('--save_suffix', type=str, default='',
                        help="Suffix to add to the output file name")
    parser.add_argument('--exp', type=str, required=True,
                        help="Attribution method to evaluate")
    parser.add_argument('--config', type=str, required=True,
                        help="Configuration of the attribution method to be used")
    parser.add_argument('--num_examples', type=int, default=1,
                        help="Number of examples to visualize. This value must not exceed the number of attributions in the smallest AggAtt bin")
    parser.add_argument('--head_pos_idx', type=int, default=0,
                        help="Position of the grid cell (zero-indexed row-wise) to visualize")

    args = parser.parse_args()
    main(args)
