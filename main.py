import os

from Pipeline import HDEPICPipeline 


if __name__ == "__main__":

    DRY_RUN = False
    FRAMES_MASKS_DIR = "../extracted_frames_and_masks"
    RECON_OUTPUT_DIR = "./recon_output"
    MASKS_ANNOTATIONS_JSON_PATH = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/hd-epic-annotations/scene-and-object-movements/mask_info.json"
    ASSOC_ANNOTATIONS_JSON_PATH = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/hd-epic-annotations/scene-and-object-movements/assoc_info.json"
    ATTACHED_MASKS_DIR = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/masks/masks"
    UNATTACHED_MASKS_DIR = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/masks/unattached_masks"

    PARTICIPANT_IDS = ["P06"]
    
    for participant_id in PARTICIPANT_IDS:
        video_dir = os.path.join("../data/hd-epic/HD-EPIC/Videos", participant_id)
        video_names = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        for video_name in video_names:
            video_name = video_name.replace(".mp4", "")
            print(f"Processing video: {video_name}")
            try:
                video_output_dir = os.path.join(FRAMES_MASKS_DIR, video_name)
                video_frames_output_dir = os.path.join(video_output_dir, "images")
                video_masks_output_dir = os.path.join(video_output_dir)
                
                pipeline = HDEPICPipeline(MASKS_ANNOTATIONS_JSON_PATH, ASSOC_ANNOTATIONS_JSON_PATH, ATTACHED_MASKS_DIR, UNATTACHED_MASKS_DIR)
                pipeline.run_pipeline(participant_id, video_name, video_frames_output_dir, video_masks_output_dir, dry_run=DRY_RUN)


            except Exception as e:
                print(f"Error processing video {video_name}: {e}")
                import traceback
                traceback.print_exc()