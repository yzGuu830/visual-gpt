"""Visual Transformer Autoregressive Decoding with Classifier-rejection"""

import imageio.v2 as imageio
from math import ceil, sqrt

import torch
import numpy as np
import argparse

from vgpt import CondVisualGPT, ClassConditionedRejectionSampler
from transformers import AutoImageProcessor, AutoModelForImageClassification



def normalize(img, min, max):
    return (img - min) / (max - min)


def make_canvas(image_size, latent_size, patch_size, apply_grids=True, grid_color=(0,0,0)):
    if not apply_grids:
        return np.ones(shape=image_size+(3,), dtype=np.uint8)*255
    else:
        i1, i2 = image_size
        l1, l2 = latent_size
        p1, p2 = patch_size
        canvas = np.ones(shape=(i1+l1-1, i2+l2-1, 3), dtype=np.uint8)*255
        for i in range(l1-1):
            idx = (i + 1) * p1 + i
            canvas[idx, :] = grid_color
        for j in range(l2-1):
            idx = (j + 1) * p2 + j
            canvas[:, idx, :] = grid_color
        return canvas

def ar_decoding_imaging(x_hat, code, z_shape, save_path="autoregressive_generation.gif", apply_grids=True):

    image_size = x_hat.size()[-2:]    
    patch_size = (image_size[0] // z_shape[0], image_size[1] // z_shape[1])
    pxl_min, pxl_max = x_hat.min(), x_hat.max()
    
    canvas = make_canvas(image_size, z_shape, patch_size, apply_grids, grid_color=(224, 224, 224))
    frames = []
    for row in range(code.size(1)):
        for col in range(code.size(2)):
            
            y0, y1 = row * patch_size[0], (row + 1) * patch_size[0]
            x0, x1 = col * patch_size[1], (col + 1) * patch_size[1]
            
            dec_patch = x_hat[0, :, y0:y1, x0:x1]
            
            if apply_grids:
                y0, y1 = y0 + row, y1 + row
                x0, x1 = x0 + col, x1 + col
            
            dec_patch = normalize(dec_patch.permute(1, 2, 0), pxl_min, pxl_max)
            
            dec_patch_np = 255 * dec_patch.cpu().numpy()
            canvas[y0:y1, x0:x1] = dec_patch_np.astype(np.uint8)

            frames.append(canvas.copy())
    
    if save_path is not None:
        imageio.mimsave(save_path, frames, format='GIF', duration=[25]*(len(frames)-1) + [2500], loop=3)
    
    return frames


def ar_decoding_imaging_multiple(x_hats: list, codes: list, z_shape: tuple, save_path: str):

    all_frames = []
    for i in range(len(x_hats)):
        fms = ar_decoding_imaging(x_hat=x_hats[i], code=codes[i], z_shape=z_shape, save_path=None)
        all_frames.append(fms)

    num_imgs = len(all_frames)
    frame_height, frame_width, _ = all_frames[0][0].shape

    max_cols = 4
    if num_imgs <= max_cols:
        rows, cols = 1, num_imgs
    else:
        cols = ceil(sqrt(num_imgs))
        rows = ceil(num_imgs / cols)

    num_missing = rows * cols - num_imgs
    if num_missing > 0:
        blank_frame = [np.zeros_like(fms[0]) for _ in range(len(fms))]
        for _ in range(num_missing):
            all_frames.append(blank_frame)

    stacked_images = []
    num_frames = len(all_frames[0])
    for i in range(num_frames):
        row_imgs = []
        for r in range(rows):
            row = [all_frames[r * cols + c][i] for c in range(cols)]
            row_imgs.append(np.concatenate(row, axis=1))
        full_image = np.concatenate(row_imgs, axis=0)
        stacked_images.append(full_image)

    if save_path is not None:
        durations = [25] * (len(stacked_images) - 1) + [2500]
        imageio.mimsave(save_path, stacked_images, format='GIF', duration=durations, loop=3)

    return stacked_images


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from Visual GPT")
    parser.add_argument(
        "--from_pretrained",
        type=str,
        default="outputs/vqgan-stfdogs",
    )
    parser.add_argument(
        "--cls_name",
        type=str,
        nargs="+",
        default="maltese",
        help="Class name for sampling images (e.g., 'maltese')"
    )
    parser.add_argument(
        "--accept_n",
        type=int,
        default=1,
        help="Choose 1 out of `accept_n` generated images based on classifier scores."
    )
    parser.add_argument(
        "--temperature",
        type=float, 
        default=1.0,
    )
    parser.add_argument(
        "--top_k",
        type=int, 
        default=100,
    )
    parser.add_argument(
        "--z_shape",
        type=lambda x: tuple(map(int, x.split(","))),
        default="16,16",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="ar_generation.gif",
        help="Path to save the generated GIF"
    )
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


if __name__ == "__main__":
    args = parse_args()

    model = CondVisualGPT.from_pretrained(args.from_pretrained)
    model.to(args.device)
    model.eval()
    if args.accept_n > 1:
        processor = AutoImageProcessor.from_pretrained("wesleyacheng/dog-breeds-multiclass-image-classification-with-vit")
        classifier = AutoModelForImageClassification.from_pretrained("wesleyacheng/dog-breeds-multiclass-image-classification-with-vit")
        classifier.to(args.device)
        processor.do_rescale = False
        rej_sampler = ClassConditionedRejectionSampler(model, classifier, processor)

    x_hats, codes = [], []
    for cls_n in args.cls_name:
        if args.accept_n == 1:
            print(f"sampling class: {cls_n}...")
            class_id = next(int(k) for k, v in model.id2label.items() if (v == cls_n or v.startswith(cls_n)))
            cond = torch.full((1, 1), class_id, dtype=torch.long, device=model.gpt.device)
            x_hat, code = model.sample(cond=cond, z_shape=args.z_shape, num_return_sequences=1, do_sample=True, temperature=args.temperature, top_k=args.top_k)
            x_hats.append(x_hat.detach().cpu())
            codes.append(code)
            print("done!")
        else:
            x_hat, code, p = rej_sampler.sample(cls_name=cls_n, accept_n=args.accept_n, do_sample=True, temperature=args.temperature, top_k=args.top_k)
            x_hats.append(x_hat)
            codes.append(code.detach().cpu())
    
    print(f"saving generated images into {args.save_path}...")
    if len(x_hats) == 1:
        ar_decoding_imaging(x_hat=x_hats[0], code=codes[0], z_shape=args.z_shape, save_path=args.save_path)
    else:
        ar_decoding_imaging_multiple(x_hats=x_hats, codes=codes, z_shape=args.z_shape, save_path=args.save_path)
    print("done!")


"""
python sample.py --from_pretrained ../outputs/vqgan-stfdogs --cls_name maltese --accept_n 3 --temperature 1.0 --top_k 100

python sample.py --from_pretrained ../outputs/vqgan-stfdogs \
    --cls_name maltese west_highland english_foxhound \
    --accept_n 1 --temperature 1.0 --top_k 100

"""
    