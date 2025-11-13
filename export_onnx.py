# INSERT_YOUR_CODE
import argparse
import torch
import os

from models import build_model
import util.misc as utils

def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer Export ONNX', add_help=False)

    # model parameters
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'))
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dim_feedforward', default=512, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--nheads', default=8, type=int)

    # loss/matcher (not used for export, but required by build_model)
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_point', default=0.05, type=float)
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)
    parser.add_argument('--point_loss_coef', default=5.0, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="./data/ShanghaiTech/PartA", type=str)

    # misc parameters
    parser.add_argument('--device', default='cpu', help='device to use for export')
    parser.add_argument('--resume', default='outputs/SHA/ckpt_251024/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--onnx_out', default='pet_exported.onnx', type=str)
    parser.add_argument('--opset', default=11, type=int)
    parser.add_argument('--img-size', nargs='+', type=int, default=[1280, 1280], help='image size')  # height, width


    return parser

def main(args):
    # Build model
    device = torch.device(args.device)
    model, _ = build_model(args)
    model.to(device)
    model.eval()

    # Load weights
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    else:
        print(f"Checkpoint not found at {args.resume}")
        return

    # Dummy input for export (Batch x 3 x H x W)
    dummy_input = torch.randn(1, 3, args.img_size[0], args.img_size[1], device=device)  # Default 512x512

    # NestedTensor preparation (test_single_image uses utils.nested_tensor_from_tensor_list)
    samples = utils.nested_tensor_from_tensor_list([dummy_input[0]])
    # Move to batch
    samples = samples.to(device)

    # Do a forward pass to check
    with torch.no_grad():
        outputs = model(samples)

    # Actual export
    input_names = ["tensors", "mask"]
    output_names = list(outputs.keys())
    torch.onnx.export(model,
                      (samples,),
                      args.onnx_out,
                      opset_version=args.opset,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes={
                          "tensors": {0: "batch", 2: "height", 3: "width"},
                          "mask": {0: "batch", 1: "height", 2: "width"},
                      },
                      do_constant_folding=True,
                      verbose=True)

    print(f"Model has been exported to {args.onnx_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET ONNX Export Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
