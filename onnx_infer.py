# INSERT_YOUR_CODE
import argparse
import numpy as np
import torch
import onnxruntime as ort
import cv2
from PIL import Image
import os
def get_args_parser():
    parser = argparse.ArgumentParser('PET ONNX Inference', add_help=False)
    parser.add_argument('--onnx_model', default='pet_exported.onnx', type=str)
    parser.add_argument('--img_path',default='vlc-record-2025-09-11-10h57m28s-rtsp___192.168.41.16_stream1-.mp4_20251027_102727.835.jpg', type=str, help='Path to input image')
    parser.add_argument('--device', default='cuda', type=str, help='Execution device for onnxruntime (cuda or cpu)')
    parser.add_argument('--img_size', nargs='+', type=int, default=[1280, 1280], help='Resize image to [h, w]')
    parser.add_argument('--vis_dir', default='vis', type=str, help='Directory to save visualization')
    return parser

def preprocess_image(img_path, img_size):
    # Read and resize
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
    # Normalize (same as train: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # To tensor: (C, H, W)
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, 0) # [1, 3, H, W]
    return img

def make_mask(img_size):
    # unmasked (all False)
    mask = np.zeros((1, img_size[0], img_size[1]), dtype=bool)
    return mask

def main(args):
    # Prepare session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.device == 'cuda' else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(args.onnx_model, providers=providers)

    # Prepare input
    img = preprocess_image(args.img_path, args.img_size)
    mask = make_mask(args.img_size)

    inputs = {
        "tensors": img.astype(np.float32),
        "mask": mask.astype(np.bool_)
    }

    # Run inference
    outputs = ort_session.run(None, inputs)
    pred_logits, pred_points, split_map_raw = outputs

    # Post-process logits: pick out object/query predictions (assuming class 1 is object)
    # pred_logits: [B, Q, C]; pred_points: [B, Q, 2]
    probs = softmax(pred_logits, axis=-1)
    scores = probs[..., 1] # class 1 is assumed as "object" or foreground
    points = pred_points[0] # [Q, 2]

    # Thresholding scores and printing number of detected points
    threshold = 0.5
    keep = scores[0] > threshold
    num_points = np.sum(keep)
    points_kept = points[keep]

    # Print results
    print(f"Total predicted points: {points.shape[0]}")
    print(f"Points with score > {threshold}: {num_points}")
    for i, (p, s) in enumerate(zip(points_kept, scores[0][keep])):
        print(f"  Point {i}: (y={p[0]:.2f}, x={p[1]:.2f}), score={s:.3f}")

    # Optional: visualize detections
    try:
        orig = cv2.imread(args.img_path)
        h0, w0 = orig.shape[:2]
        h1, w1 = args.img_size
        scale_y = h0 / h1
        scale_x = w0 / w1
        for p in points_kept:
            cy = int(p[0]*h1*scale_y)
            cx = int(p[1]*w1*scale_x)
            cv2.circle(orig, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(orig, f'prediction: {num_points}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
        cv2.imwrite(os.path.join(args.vis_dir, f'{args.img_path.split(".")[0]}_pred{num_points}.jpg'), orig)
    except Exception as e:
        print("Visualization skipped:", e)

def softmax(x, axis=None):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET ONNX Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
