import os
from pathlib import Path

from ExtractionPipeline import HDEPICExtractionPipeline
from ReconstructionPipeline import HDEPIC3DReconstructionPipeline

class HDEPICPipeline:
    def __init__(self, masks_annotation_json_path, assoc_annotation_json_path, attached_masks_dir, unattached_masks_dir):
        self.extraction_pipeline = HDEPICExtractionPipeline(masks_annotation_json_path, assoc_annotation_json_path, attached_masks_dir, unattached_masks_dir)
        self.reconstruction_pipeline = HDEPIC3DReconstructionPipeline()

    def run_pipeline(self, participant_id, video_name, frames_output_folder, masks_output_dir, resize_to=None, dry_run=False, skip_step_1=False):
        
        if self._check_reconstruction_complete(video_name):
            print(f"Reconstruction for video {video_name} already complete, skipping pipeline execution.")
            return True
        
        # Step 1: Extract frames and masks
        if not skip_step_1:
            self.extraction_pipeline.extract_frames(participant_id, video_name, frames_output_folder, resize_to=resize_to, dry_run=dry_run)
            self.extraction_pipeline.extract_video_masks(video_name, frames_output_folder, masks_output_dir, dry_run=dry_run)

        # Step 2: Reconstruct 3D from extracted frames and masks
        self.reconstruction_pipeline.reconstruct_3d(video_name, masks_output_dir, resize_to=resize_to, dry_run=dry_run)

        return True
    
    def _check_reconstruction_complete(self, video_name):
        viz_base_dir = os.path.join(Path(__file__).parent, "visualization") # the viz dir should be in the same 'pipeline' dir as this file
        participant_viz_dir = os.path.join(viz_base_dir, video_name)

        # check if participant viz dir is non-empty
        # TODO: this doesnt guarantee that ALL reconstructions for all objects are done, just that there exists some reconstructions
        if os.path.exists(participant_viz_dir) and os.listdir(participant_viz_dir):
            return True
        return False


if __name__ == "__main__":
    
    # for testing
    DRY_RUN = False
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
    
    pipeline = HDEPICPipeline(MASKS_ANNOTATIONS_JSON_PATH, ASSOC_ANNOTATIONS_JSON_PATH, ATTACHED_MASKS_DIR, UNATTACHED_MASKS_DIR)
    pipeline.run_pipeline(PARTICIPANT_ID, VIDEO_NAME, video_frames_output_dir, video_masks_output_dir, dry_run=DRY_RUN, skip_step_1=True)

    print("Pipeline execution completed.")