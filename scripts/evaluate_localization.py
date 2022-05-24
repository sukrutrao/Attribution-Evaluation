import torch
import argparse
import os
import numpy as np
from attribution_evaluation.evaluation import localization
from attribution_evaluation.models import settings
import torchvision


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    test_data_dict = torch.load(os.path.join(args.dataset_path, 'test.pt'))
    scale = test_data_dict["scale"]
    img_dims = test_data_dict["input_dims"][1:]

    attributions = torch.load(os.path.join(args.attributions_path, 'attributions_' +
                                           args.model + '_' + args.setting + '_' + os.path.basename(args.dataset_path) + '_' + args.exp + '_' + args.config + '_' + str(args.layer) + args.save_suffix + '.pt'))

    localization_scores = localization.get_localization_score(
        attributions, only_corners=settings.eval_only_corners(args.setting), img_dims=img_dims, scale=scale)
    print("Localization Scores:")
    print("Number of data points:", len(localization_scores))
    print("Mean:", "{:.4f}".format(localization_scores.mean()))
    print("Standard Deviation:", "{:.4f}".format(localization_scores.std()))
    print("Median:", "{:.4f}".format(localization_scores.median()))
    print("Min:", "{:.4f}".format(localization_scores.min()))
    print("Max:", "{:.4f}".format(localization_scores.max()))
    fig, _ = localization.plot_localization_scores_single(
        localization_scores.tolist(), args.model, args.setting, args.exp, args.config, args.layer, scale=scale)
    full_save_path = os.path.join(args.save_path, "localization_" +
                                  args.model + "_" + args.setting + "_" + os.path.basename(args.dataset_path) + '_' + args.exp + '_' + args.config + '_' + str(args.layer) + args.save_suffix + ".png")
    print("Saving box plot at", full_save_path)
    fig.savefig(full_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes the localization scores for a set of attributions and plots them on a box plot.")
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
                        help="Path of directory in which to save plot")
    parser.add_argument('--save_suffix', type=str, default='',
                        help="Suffix to add to the output file name")
    parser.add_argument('--exp', type=str, required=True,
                        help="Attribution method to evaluate")
    parser.add_argument('--config', type=str, required=True,
                        help="Configuration of the attribution method to be used")

    args = parser.parse_args()
    main(args)
