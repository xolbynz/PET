# INSERT_YOUR_REWRITE_HERE
import argparse
import torch
import os

from models import build_model
import util.misc as utils

class PETONNXWrapper(torch.nn.Module):
    """
    Inference-only wrapper for PET, exposing the signature (tensors, mask) -> tensors suitable for ONNX export.
    This is based on the logic in test_single_image.py for the inference and output processing steps.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, tensors, mask):
        # tensors: (B, 3, H, W)
        # mask: (B, H, W)
        samples = utils.NestedTensor(tensors, mask)
        outputs = self.base_model(samples, test=True)
        # Extract outputs as done in test_single_image.py
        logits = outputs['pred_logits']
        pred_points = outputs['pred_points']
        # Optionally, export the split_map_raw as well for downstream use
        split_map_raw = outputs.get('split_map_raw', None)
        return logits, pred_points, split_map_raw

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
    parser.add_argument('--device', default='cuda:0', help='device to use for export')
    parser.add_argument('--resume', default='outputs/SHA/ckpt_251024/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--onnx_out', default='pet_exported.onnx', type=str)
    parser.add_argument('--opset', default=11, type=int)
    parser.add_argument('--img-size', nargs='+', type=int, default=[1280, 1280], help='image size')
    return parser

def main(args):
    # Build model
    device = torch.device(args.device)
    model, _ = build_model(args)
    model.to(device)
    model.eval()

    # Load weights
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
    else:
        print(f"Checkpoint not found at {args.resume}")
        return

    # Compose the wrapper for ONNX export
    onnx_model = PETONNXWrapper(model).to(device)
    onnx_model.eval()

    # Dummy input (Batch, 3, H, W), mask (Batch, H, W)
    dummy_tensors = torch.randn(1, 3, args.img_size[0], args.img_size[1], device=device)
    dummy_mask = torch.zeros(1, args.img_size[0], args.img_size[1], dtype=torch.bool, device=device)  # unmasked

    # Forward sanity check
    with torch.no_grad():
        out = onnx_model(dummy_tensors, dummy_mask)
        print("Dummy forward output shapes:",
              [x.shape if x is not None else None for x in out])

    # ONNX export
    input_names = ["tensors", "mask"]
    output_names = ["pred_logits", "pred_points", "split_map_raw"]
    dynamic_axes = {
        'tensors': {0: 'batch', 2: 'height', 3: 'width'},
        'mask': {0: 'batch', 1: 'height', 2: 'width'},
        'pred_logits': {0: 'batch', 1: 'num_queries'},   # shape [B, Q, C]
        'pred_points': {0: 'batch', 1: 'num_queries'},   # shape [B, Q, 2]
        'split_map_raw': {0: 'batch', 2: 'height_out', 3: 'width_out'},  # [B, 1, H', W']
    }

    torch.onnx.export(
        onnx_model,
        (dummy_tensors, dummy_mask),
        args.onnx_out,
        opset_version=args.opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        verbose=True,
    )

    print(f"Model has been exported to {args.onnx_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET ONNX Export Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
