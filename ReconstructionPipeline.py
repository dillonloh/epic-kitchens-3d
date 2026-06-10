from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] # get to dissertation project root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MV_SAM3D")) # add MV_SAM3D to path so we can import their run_inference script
sys.path.insert(0, str(ROOT / "MV_SAM3D" / "notebook")) 

from MV_SAM3D.run_inference_weighted import *

class HDEPIC3DReconstructionPipeline:
    def __init__(self):
        pass

    def _get_all_mask_prompts(self, input_dir):
        print(f"Getting all mask prompts from input directory: {input_dir}")
        mask_prompts = []
        for directory in os.listdir(input_dir):
            if directory == "images": # skip the images directory
                continue

            if os.path.isdir(os.path.join(input_dir, directory)): # only consider directories (which should be the mask prompt directories)
                mask_prompts.append(directory)
        
        return mask_prompts

    def reconstruct_3d(self, video_name, input_dir, mask_prompt=None, resize_to=None, dry_run=False):
        
        print(f"Reconstructing 3D from video: {video_name}")

        # we follow the original MV-SAM3D's run_inference script arg signature (using their default values too)
        # TODO: switch to using their lower level API instead of the run_inference script, to avoid having to do this hacky arg parsing
        mask_prompts = self._get_all_mask_prompts(input_dir) if mask_prompt is None else [mask_prompt]
        print(f"Using mask prompts: {mask_prompts}")

        for single_mask_prompt in mask_prompts:
            # skip if the mask prompt contains the word "skipped"
            if "skipped" in single_mask_prompt:
                print(f"Skipping mask prompt {single_mask_prompt} for video {video_name} because it contains the word 'skipped'.")
                continue

            image_names = None
            model_tag = "hf"

            seed = 42
            stage1_steps = 50
            stage2_steps = 25
            decode_formats = ["gaussian", "mesh"]

            # Stage 1 (Shape) Weighting Parameters
            no_stage1_weighting = False
            stage1_entropy_layer = 9
            stage1_entropy_alpha = 30.0

            # Stage 2 (Texture) Weighting Parameters
            no_stage2_weighting = False
            stage2_weight_source = "entropy"
            stage2_entropy_alpha = 30.0
            stage2_visibility_alpha = 30.0
            stage2_attention_layer = 6
            stage2_attention_step = 0
            stage2_min_weight = 0.001
            stage2_weight_combine_mode = "average"
            stage2_visibility_weight_ratio = 0.5

            # Visualization Parameters
            visualize_weights = False
            save_attention = False
            attention_layers = None
            save_stage2_init = False

            # DA3 Integration Parameters
            da3_output = None
            merge_da3_glb = False
            overlay_pointmap = False
            compute_latent_visibility = False
            self_occlusion_tolerance = 4.0

            # Pose Optimization Parameters
            run_pose_optimization = False
            pose_opt_iterations = 300
            pose_opt_lr = 0.01
            pose_opt_mask_erosion = 3
            pose_opt_device = "cuda"
            pose_opt_optimize_scale = False

            input_path = Path(input_dir)
            if not input_path.exists():
                raise FileNotFoundError(f"Input path does not exist: {input_path}")

            image_names = parse_image_names(image_names)

            # check if reconstruction for this object already exists, and if so skip it (this allows us to run the pipeline multiple times without re-reconstructing objects we've already done, in case of crashes or if we want to add more objects later)
            if self._check_object_reconstruction_complete(video_name, single_mask_prompt):
                print(f"Reconstruction for object {single_mask_prompt} in video {video_name} already exists, skipping.")
                continue

            try:
                logger.info(f"Single-object mode: {single_mask_prompt}")
                run_weighted_inference(
                    input_path=input_path,
                    mask_prompt=single_mask_prompt,
                    image_names=image_names,
                    seed=seed,
                    stage1_steps=stage1_steps,
                    stage2_steps=stage2_steps,
                    decode_formats=decode_formats,
                    model_tag=model_tag,
                    stage1_weighting=not no_stage1_weighting,
                    stage1_entropy_layer=stage1_entropy_layer,
                    stage1_entropy_alpha=stage1_entropy_alpha,
                    stage2_weighting=not no_stage2_weighting,
                    stage2_weight_source=stage2_weight_source,
                    stage2_entropy_alpha=stage2_entropy_alpha,
                    stage2_visibility_alpha=stage2_visibility_alpha,
                    stage2_attention_layer=stage2_attention_layer,
                    stage2_attention_step=stage2_attention_step,
                    stage2_min_weight=stage2_min_weight,
                    stage2_weight_combine_mode=stage2_weight_combine_mode,
                    stage2_visibility_weight_ratio=stage2_visibility_weight_ratio,
                    visualize_weights=visualize_weights,
                    save_attention=save_attention,
                    attention_layers_to_save=parse_attention_layers(attention_layers),
                    save_stage2_init=save_stage2_init,
                    da3_output_path=da3_output,
                    merge_da3_glb=merge_da3_glb,
                    overlay_pointmap=overlay_pointmap,
                    enable_latent_visibility=compute_latent_visibility,
                    self_occlusion_tolerance=self_occlusion_tolerance,
                    run_pose_optimization=run_pose_optimization,
                    pose_opt_iterations=pose_opt_iterations,
                    pose_opt_lr=pose_opt_lr,
                    pose_opt_mask_erosion=pose_opt_mask_erosion,
                    pose_opt_device=pose_opt_device,
                    pose_opt_optimize_scale=pose_opt_optimize_scale,
                )


            except Exception as e:
                print(f"Error during 3D reconstruction for video {video_name}: {e}")
                
                import traceback
                traceback.print_exc()
                
                return None
            
        return True

    def _check_object_reconstruction_complete(self, video_name, mask_prompt):
        object_viz_dir = Path(__file__).parent / "visualization" / video_name / mask_prompt

        if not object_viz_dir.exists():
            print(f"Object visualization directory does not exist: {object_viz_dir}")
            return False

        for run_dir in object_viz_dir.iterdir():
            if not run_dir.is_dir():
                continue

            has_mesh = (run_dir / "result.glb").exists()
            has_params = (run_dir / "params.npz").exists()
            if has_mesh and has_params:
                print(f"Found complete reconstruction for object {mask_prompt} in video {video_name} at {run_dir}, skipping reconstruction for this object.")
                return True

        print(f"No complete reconstruction found for object {mask_prompt} in video {video_name}, will reconstruct.")
        return False

if __name__ == "__main__":
    
    # for testing
    DRY_RUN = False
    PARTICIPANT_ID = "P01"
    VIDEO_NAME = "P01-20240202-110250"
    FRAMES_MASKS_DIR = "./extracted_frames_and_masks"
    INPUT_DIR = os.path.join(FRAMES_MASKS_DIR, VIDEO_NAME)
    OUTPUT_DIR = os.path.join("./3d_reconstructions", VIDEO_NAME)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline = HDEPIC3DReconstructionPipeline()
    output_dir = pipeline.reconstruct_3d(VIDEO_NAME, INPUT_DIR, OUTPUT_DIR, dry_run=DRY_RUN)
