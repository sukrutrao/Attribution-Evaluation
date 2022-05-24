import torch
from attribution_evaluation.attribution import attributors
import argparse
import os
import numpy as np
from attribution_evaluation.models import models, settings
from tqdm import tqdm


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    test_data_dict = torch.load(os.path.join(args.dataset_path, 'test.pt'))

    test_data = torch.utils.data.TensorDataset(
        test_data_dict["data"], test_data_dict["labels"])

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False)
    scale = test_data_dict["scale"]
    grid_size = scale * scale

    model = models.get_model(args.model)()
    model_setting = settings.get_setting(args.setting)(model=model, scale=scale)
    if args.cuda:
        model_setting.cuda()
    model_setting.eval()

    if not settings.eval_only_corners(args.setting):
        head_list = [0]
    else:
        head_list = [0, grid_size - 1]

    attributor = attributors.AttributorContainer(
        model_setting=model_setting, base_exp=args.exp, base_config=args.config)

    attributions = []
    for (test_X, test_y) in tqdm(test_loader):
        if args.cuda:
            test_X = test_X.cuda().requires_grad_(True)
            test_y = test_y.cuda()
        batch_attributions = []
        for head_pos_idx in head_list:
            if model_setting.single_head:
                batch_attributions.append(
                    attributor.attribute_selection(img=test_X, target=test_y, conv_layer_idx=args.layer).sum(dim=2, keepdim=True))
            else:
                batch_attributions.append(attributor.attribute_selection(img=test_X, target=test_y[:, head_pos_idx].reshape(
                    -1, 1), output_head_idx=head_pos_idx, conv_layer_idx=args.layer).sum(dim=2, keepdim=True))
        attributions.append(torch.cat(batch_attributions, dim=1))
    attributions = torch.cat(attributions, dim=0).detach().cpu()
    full_save_path = os.path.join(args.save_path, 'attributions_' +
                                  args.model + '_' + args.setting + '_' + os.path.basename(args.dataset_path) + '_' + args.exp + '_' + args.config + '_' + str(args.layer) + args.save_suffix + '.pt')
    print("Saving attributions at", full_save_path)
    torch.save(attributions, full_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs an attribution method using a specified configuration at a specified layer.")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Path of directory containing the dataset")
    parser.add_argument('--cuda', action='store_true',
                        default=False, help="Flag to enable running on GPU")
    parser.add_argument('--seed', type=int, default=1, help="Random seed value")
    parser.add_argument('--model', type=str, required=True,
                        choices=["vgg11", "resnet18"], help="Model to evaluate on")
    parser.add_argument('--setting', type=str, required=True,
                        choices=["GridPG", "DiFull", "DiPart"], help="Setting to evaluate on")
    parser.add_argument('--layer', type=str, required=True, choices=["Input", "Middle", "Final"],
                        help="Layer to evaluate on")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for inputs during evaluation")
    parser.add_argument('--save_path', type=str, required=True,
                        help="Path of directory in which to save attributions")
    parser.add_argument('--save_suffix', type=str, default='',
                        help="Suffix to add to the output file name")
    parser.add_argument('--exp', type=str, required=True,
                        help="Attribution method to evaluate")
    parser.add_argument('--config', type=str, required=True,
                        help="Configuration of the attribution method to be used")

    args = parser.parse_args()
    main(args)
