import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics

import torch
import numpy as np
from PIL import Image

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *
import os
import json


def translate_and_restore(image, shift):
    """
    Translate the image horizontally to the right by `shift` pixels.
    The left `shift` columns are kept as in the original image ("restored").
    Implementation:
      - for columns [shift, W-1], we take pixels from [0, W-shift-1]
      - for columns [0, shift-1], we keep the original columns
    """
    arr = np.array(image)
    H, W = arr.shape[:2]
    if shift <= 0:
        return image
    shift = shift % W

    attacked = arr.copy()
    # columns [shift, W) get original [0, W-shift)
    attacked[:, shift:, ...] = arr[:, :W-shift, ...]
    # columns [0, shift) remain as in original (already in copy)
    return Image.fromarray(attacked)


def main(args):
    table = None
    # base directory for this run (used for images + results.json)
    base_dir = os.path.join(args.image_out_dir, args.run_name)
    os.makedirs(base_dir, exist_ok=True)

    clean_dir = watermarked_dir = attacked_clean_dir = attacked_wm_dir = None
    if args.save_images:
        clean_dir = os.path.join(base_dir, 'clean')
        watermarked_dir = os.path.join(base_dir, 'watermarked')
        attacked_clean_dir = os.path.join(base_dir, 'attacked_clean')
        attacked_wm_dir = os.path.join(base_dir, 'attacked_watermarked')

        for d in [clean_dir, watermarked_dir, attacked_clean_dir, attacked_wm_dir]:
            os.makedirs(d, exist_ok=True)

    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark'])
        wandb.config.update(args)
        table = wandb.Table(columns=[
            'gen_no_w', 'no_w_clip_score',
            'gen_w', 'w_clip_score',
            'prompt', 'no_w_metric', 'w_metric'
        ])
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.model_id,
        subfolder='scheduler'
    )
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
    )
    pipe = pipe.to(device)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model,
            pretrained=args.reference_model_pretrain,
            device=device
        )
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    tester_prompt = ''  # assume at detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # ground-truth patch
    gt_patch = get_watermarking_pattern(pipe, args, device)

    results = []
    clip_scores = []
    clip_scores_w = []

    # scores used for ROC (higher = more likely watermarked)
    no_w_scores = []       # scores for clean images
    w_scores = []          # scores for watermarked images
    attacked_scores = []   # scores for attacked watermarked images

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        
        current_prompt = dataset[i][prompt_key]
        
        ### generation
        # generation without watermarking
        set_random_seed(seed)
        init_latents_no_w = pipe.get_random_latents()
        outputs_no_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_no_w,
        )
        orig_image_no_w = outputs_no_w.images[0]
        
        # generation with watermarking
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

        # inject watermark
        init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args)

        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
        )
        orig_image_w = outputs_w.images[0]

        # attacked images (translation + restoration)
        attacked_clean_img = translate_and_restore(orig_image_no_w, args.attack_shift)
        attacked_wm_img = translate_and_restore(orig_image_w, args.attack_shift)

        # save images if requested
        if args.save_images:
            idx_str = f"{i:06d}"

            # original clean & watermarked
            clean_path = os.path.join(clean_dir, f"{idx_str}_clean.png")
            wm_path = os.path.join(watermarked_dir, f"{idx_str}_watermarked.png")
            orig_image_no_w.save(clean_path)
            orig_image_w.save(wm_path)

            # attacked / distorted versions
            attacked_clean_path = os.path.join(attacked_clean_dir, f"{idx_str}_attacked_clean.png")
            attacked_wm_path = os.path.join(attacked_wm_dir, f"{idx_str}_attacked_watermarked.png")
            attacked_clean_img.save(attacked_clean_path)
            attacked_wm_img.save(attacked_wm_path)

        ### detection pipeline (no extra augmentations)

        # reverse clean image
        img_no_w = transform_img(orig_image_no_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)
        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # reverse watermarked image
        img_w = transform_img(orig_image_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)
        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # reverse attacked watermarked image
        img_attacked = transform_img(attacked_wm_img).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_attacked = pipe.get_image_latents(img_attacked, sample=False)
        reversed_latents_attacked = pipe.forward_diffusion(
            latents=image_latents_attacked,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # eval for clean vs watermarked
        no_w_metric, w_metric = eval_watermark(
            reversed_latents_no_w,
            reversed_latents_w,
            watermarking_mask,
            gt_patch,
            args
        )

        # eval for attacked image: reuse eval_watermark, second output is for attacked
        _, attacked_metric = eval_watermark(
            reversed_latents_no_w,
            reversed_latents_attacked,
            watermarking_mask,
            gt_patch,
            args
        )

        # similarity (optional)
        if args.reference_model is not None:
            sims = measure_similarity(
                [orig_image_no_w, orig_image_w],
                current_prompt,
                ref_model,
                ref_clip_preprocess,
                ref_tokenizer,
                device
            )
            w_no_sim = sims[0].item()
            w_sim = sims[1].item()
        else:
            w_no_sim = 0.0
            w_sim = 0.0

        # score = negative of metric (higher = more watermarked)
        score_no_w = -float(no_w_metric)
        score_w = -float(w_metric)
        score_attacked = -float(attacked_metric)

        no_w_scores.append(score_no_w)
        w_scores.append(score_w)
        attacked_scores.append(score_attacked)

        results.append({
            'index': int(i),
            'seed': int(seed),
            'prompt': current_prompt,
            'no_w_metric': float(no_w_metric),
            'w_metric': float(w_metric),
            'attacked_metric': float(attacked_metric),
            'no_w_score': score_no_w,
            'w_score': score_w,
            'attacked_score': score_attacked,
            'w_no_sim': float(w_no_sim),
            'w_sim': float(w_sim),
        })

        if args.with_tracking:
            if (args.reference_model is not None) and (i < args.max_num_log_image):
                # log images when we use reference_model
                table.add_data(
                    wandb.Image(orig_image_no_w), w_no_sim,
                    wandb.Image(orig_image_w), w_sim,
                    current_prompt, no_w_metric, w_metric
                )
            else:
                table.add_data(
                    None, w_no_sim,
                    None, w_sim,
                    current_prompt, no_w_metric, w_metric
                )

            clip_scores.append(w_no_sim)
            clip_scores_w.append(w_sim)

    # ---- ROC + threshold from clean vs watermarked ----
    preds = no_w_scores + w_scores
    t_labels = [0] * len(no_w_scores) + [1] * len(w_scores)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)

    # TPR at ~1% FPR + corresponding threshold
    if np.any(fpr < 0.01):
        idx = np.where(fpr < 0.01)[0][-1]
    else:
        idx = np.argmin(np.abs(fpr - 0.01))
    low = tpr[idx]  # TPR@1%FPR
    thresh = float(thresholds[idx])

    # ---- Apply threshold to clean / watermarked / attacked ----
    no_w_scores_arr = np.array(no_w_scores)
    w_scores_arr = np.array(w_scores)
    attacked_scores_arr = np.array(attacked_scores)

    pred_clean = no_w_scores_arr >= thresh
    pred_wm = w_scores_arr >= thresh
    pred_attacked = attacked_scores_arr >= thresh

    fpr_at_thresh = pred_clean.mean()  # since labels for clean are 0
    tpr_at_thresh = pred_wm.mean()     # TPR for original watermarked
    tpr_attacked = pred_attacked.mean()  # detection rate on attacked images
  
      # ---- Attach per-image detection flags into results ----
    for idx, r in enumerate(results):
        r['clean_detected'] = bool(pred_clean[idx])
        r['watermarked_detected'] = bool(pred_wm[idx])
        r['attacked_detected'] = bool(pred_attacked[idx])

    # ---- Save results as JSON ----
    results_json = {
        "config": vars(args),
        "overall_metrics": {
            "auc": float(auc),
            "acc": float(acc),
            "TPR@1%FPR": float(low),
            "threshold_for_1%FPR": float(thresh),
            "FPR_on_clean_at_thresh": float(fpr_at_thresh),
            "TPR_on_watermarked_at_thresh": float(tpr_at_thresh),
            "TPR_on_attacked_at_thresh": float(tpr_attacked),
        },
        "num_samples": len(results),
        "per_sample_results": results,
    }

    json_path = os.path.join(base_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({
            'clip_score_mean': mean(clip_scores) if clip_scores else 0.0,
            'clip_score_std': stdev(clip_scores) if len(clip_scores) > 1 else 0.0,
            'w_clip_score_mean': mean(clip_scores_w) if clip_scores_w else 0.0,
            'w_clip_score_std': stdev(clip_scores_w) if len(clip_scores_w) > 1 else 0.0,
            'auc': auc,
            'acc': acc,
            'TPR@1%FPR': low,
            'FPR_on_clean_at_thresh': fpr_at_thresh,
            'TPR_on_watermarked_at_thresh': tpr_at_thresh,
            'TPR_on_attacked_at_thresh': tpr_attacked,
        })
    
    print(f'auc: {auc:.4f}, acc: {acc:.4f}, TPR@1%FPR: {low:.4f}')
    print(f'Threshold (score) at ~1% FPR: {thresh:.4f}')
    print(f'FPR on clean at threshold: {fpr_at_thresh:.4f}')
    print(f'TPR on watermarked at threshold: {tpr_at_thresh:.4f}')
    print(f'TPR on attacked watermarked at threshold: {tpr_attacked:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='prompts_100.txt')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--image_out_dir', default='saved_images', type=str)
    parser.add_argument('--attack_shift', default=40, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    # (kept for compatibility, but not used now for augmentations)
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)
