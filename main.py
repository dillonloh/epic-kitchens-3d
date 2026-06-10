import argparse
import os
from pathlib import Path

from Pipeline import HDEPICPipeline 

RECONSTRUCTION_TRACKING_LOG = Path(__file__).parent / "reconstruction_tracking_log.txt" # log to keep track of which videos have been reconstructed, in case we want to run the pipeline multiple times and skip already reconstructed videos

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='pipeline',
                    description='extract shit',)
    
    parser.add_argument('--video_name', required=True, type=str)
    
    args = parser.parse_args()

    video_name = args.video_name
    participant_id = video_name.split('-')[0]

    DRY_RUN = False
    FRAMES_MASKS_DIR = "./extracted_frames_and_masks"
    RECON_OUTPUT_DIR = "./recon_output"
    MASKS_ANNOTATIONS_JSON_PATH = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/hd-epic-annotations/scene-and-object-movements/mask_info.json"
    ASSOC_ANNOTATIONS_JSON_PATH = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/hd-epic-annotations/scene-and-object-movements/assoc_info.json"
    ATTACHED_MASKS_DIR = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/masks/masks"
    UNATTACHED_MASKS_DIR = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/masks/unattached_masks"

    video_name = video_name.replace(".mp4", "")
    print(f"Processing video: {video_name}")

    try:
        # check if this video has already been reconstructed by looking in the reconstruction tracking log
        if os.path.exists(RECONSTRUCTION_TRACKING_LOG):
            with open(RECONSTRUCTION_TRACKING_LOG, "r") as log_file:
                reconstructed_videos = log_file.read().splitlines()
                if video_name in reconstructed_videos:
                    print(f"Video {video_name} already reconstructed, skipping...")
                    exit()
    
        video_output_dir = os.path.join(FRAMES_MASKS_DIR, video_name)
        video_frames_output_dir = os.path.join(video_output_dir, "images")
        video_masks_output_dir = os.path.join(video_output_dir)
        
        pipeline = HDEPICPipeline(MASKS_ANNOTATIONS_JSON_PATH, ASSOC_ANNOTATIONS_JSON_PATH, ATTACHED_MASKS_DIR, UNATTACHED_MASKS_DIR)
        pipeline.run_pipeline(participant_id, video_name, video_frames_output_dir, video_masks_output_dir, dry_run=DRY_RUN)

        # log the reconstruction
        with open(RECONSTRUCTION_TRACKING_LOG, "a") as log_file:
            log_file.write(f"{video_name}\n")

    except Exception as e:
        print(f"Error processing video {video_name}: {e}")
        import traceback
        traceback.print_exc()
