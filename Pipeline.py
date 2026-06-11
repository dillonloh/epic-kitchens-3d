import os
import shutil
from pathlib import Path
from typing import List

from ExtractionPipeline import HDEPICExtractionPipeline
from ReconstructionPipeline import HDEPIC3DReconstructionPipeline

class HDEPICPipeline:
    def __init__(self, masks_annotation_json_path, assoc_annotation_json_path, attached_masks_dir, unattached_masks_dir, reconstruction_tracking_log_path):
        self.extraction_pipeline = HDEPICExtractionPipeline(masks_annotation_json_path, assoc_annotation_json_path, attached_masks_dir, unattached_masks_dir)
        self.reconstruction_pipeline = HDEPIC3DReconstructionPipeline()
        
        # log to keep track of which videos have been reconstructed, in case we want to run the pipeline multiple times and skip already reconstructed videos
        self.reconstruction_tracking_log = reconstruction_tracking_log_path

    def run_pipeline(self, participant_id, video_name, frames_output_folder, masks_output_dir, resize_to=None, dry_run=False, skip_step_1=False, post_cleanup=True):
        
        if not self._check_video_reconstruction_complete(video_name):
            print(f"Running pipeline for video {video_name}...")

            # Step 1: Extract frames and masks
            if not skip_step_1:
                self.extraction_pipeline.extract_frames(participant_id, video_name, frames_output_folder, resize_to=resize_to, dry_run=dry_run)
                self.extraction_pipeline.extract_video_masks(video_name, frames_output_folder, masks_output_dir, dry_run=dry_run)

            # Step 2: Reconstruct 3D from extracted frames and masks
            self.reconstruction_pipeline.reconstruct_3d(video_name, masks_output_dir, resize_to=resize_to, dry_run=dry_run)

            print(f"Pipeline execution for video {video_name} completed.")

        else:
            print(f"Skipping pipeline execution for video {video_name} since reconstruction is already complete.")

        # Final Step: Cleanup extracted frames and masks to save space (optional, can comment out if you want to keep them)
        if not dry_run and post_cleanup:
            try:
                print(f"Cleaning up extracted frames and masks for video {video_name}...")
                # remove extracted frames
                shutil.rmtree(frames_output_folder, ignore_errors=True)
                # remove extracted masks
                shutil.rmtree(masks_output_dir, ignore_errors=True)

                with open(self.reconstruction_tracking_log, "a") as log_file:
                    log_file.write(f"{video_name}\n")

                print(f"Cleanup completed for video {video_name}.")
            except Exception as e:
                print(f"Error during cleanup for video {video_name}: {e}")
                import traceback
                traceback.print_exc()

        return True
    
    def _check_video_reconstruction_complete(self, video_name):

        if os.path.exists(self.reconstruction_tracking_log):
            with open(self.reconstruction_tracking_log, "r") as log_file:
                reconstructed_videos = log_file.read().splitlines()
                if video_name in reconstructed_videos:
                    print(f"Video {video_name} already reconstructed, skipping...")
                    return True

        mask_prompts = self._get_all_mask_prompts(video_name)

        if not mask_prompts:
            print(f"No mask prompts found in annotations for video {video_name}.")
            return False

        for mask_prompt in mask_prompts:
            if not self._check_object_reconstruction_complete(video_name, mask_prompt):
                print(f"Reconstruction for video {video_name} is not complete, missing reconstruction for object {mask_prompt}.")
                return False
        
        print(f"Reconstruction for video {video_name} is already complete, skipping reconstruction.")
        return True

    def _get_all_mask_prompts(self, video_name: str) -> List[str]:
        print(f"Getting all mask prompts from annotations for video: {video_name}")
        video_associations = self.extraction_pipeline.assocs_annotations.get(video_name, {})

        all_prompts = set()
        for _, association_details in video_associations.items():
            association_name = association_details.get("name", "")
            if "Track" in association_name or "skipped" in association_name.lower():
                continue
            all_prompts.add(association_name)

        return list(all_prompts)
    
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
    DRY_RUN = True
    PARTICIPANT_ID = "P01"
    VIDEO_NAME = "P01-20240202-110250"
    FRAMES_MASKS_DIR = "./extracted_frames_and_masks"
    RECON_OUTPUT_DIR = "./recon_output"

    MASKS_ANNOTATIONS_JSON_PATH = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/hd-epic-annotations/scene-and-object-movements/mask_info.json"
    ASSOC_ANNOTATIONS_JSON_PATH = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/hd-epic-annotations/scene-and-object-movements/assoc_info.json"
    ATTACHED_MASKS_DIR = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/masks/masks"
    UNATTACHED_MASKS_DIR = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/masks/unattached_masks"

    # TODO: move this into the pipeline logic, just pass video_output_dir
    video_output_dir = os.path.join(FRAMES_MASKS_DIR, VIDEO_NAME)
    video_frames_output_dir = os.path.join(video_output_dir, "images")
    video_masks_output_dir = os.path.join(video_output_dir)
    
    print("Starting pipeline execution...")
    pipeline = HDEPICPipeline(MASKS_ANNOTATIONS_JSON_PATH, ASSOC_ANNOTATIONS_JSON_PATH, ATTACHED_MASKS_DIR, UNATTACHED_MASKS_DIR)
    pipeline.run_pipeline(PARTICIPANT_ID, VIDEO_NAME, video_frames_output_dir, video_masks_output_dir, dry_run=DRY_RUN, skip_step_1=False, post_cleanup=True)

    print("Pipeline execution completed.")