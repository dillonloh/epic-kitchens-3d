from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
from typing import List, Optional, Set

import cv2
import numpy as np
from PIL import Image

# load scene and object movements
MASKS_ANNOTATIONS_JSON_PATH = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/hd-epic-annotations/scene-and-object-movements/mask_info.json"
ASSOC_ANNOTATIONS_JSON_PATH = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/hd-epic-annotations/scene-and-object-movements/assoc_info.json"
ATTACHED_MASKS_DIR = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/masks/masks"
UNATTACHED_MASKS_DIR = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/masks/unattached_masks"
VIDEOS_DIR = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/HD-EPIC/Videos"
EXTRACTED_FRAMES_AND_MASKS_ROOT_FOLDER = "./extracted_frames_and_masks/"

with open(MASKS_ANNOTATIONS_JSON_PATH, "r") as f:
    masks = json.load(f)

with open(ASSOC_ANNOTATIONS_JSON_PATH, "r") as f:
    associations = json.load(f)

class HDEPICExtractionPipeline:
    def __init__(self, masks_annotation_json_path, assoc_annotation_json_path, attached_masks_dir, unattached_masks_dir):
        self.masks_annotation_json_path = masks_annotation_json_path
        self.assoc_annotation_json_path = assoc_annotation_json_path
        self.attached_masks_dir = attached_masks_dir
        self.unattached_masks_dir = unattached_masks_dir
        
        self.masks_annotations = None
        self.assocs_annotations = None


        self._load_masks_annotations()
        self._load_associations_annotations()

    def _load_masks_annotations(self):
        with open(self.masks_annotation_json_path, "r") as f:
            self.masks_annotations = json.load(f)

    def _load_associations_annotations(self):
        with open(self.assoc_annotation_json_path, "r") as f:
            self.assocs_annotations = json.load(f)

    def _get_required_frame_numbers(self, video_name: str) -> List[int]:
        video_associations = self.assocs_annotations.get(video_name, {})
        video_masks = self.masks_annotations.get(video_name, {})

        required_frames = set()

        for _, association_details in video_associations.items():
            association_name = association_details.get("name", "")
            if "Track" in association_name or "skipped" in association_name.lower():
                continue

            for track in association_details.get("tracks", []):
                for mask_id in track.get("masks", []):
                    mask_info = video_masks.get(mask_id)
                    if not mask_info:
                        continue

                    frame_num = mask_info.get("frame_number")
                    if frame_num is None:
                        continue

                    required_frames.add(int(frame_num))

        return sorted(required_frames)

    def extract_frames(
        self,
        participant_id,
        video_name,
        output_folder,
        resize_to=None,
        dry_run=False,
        force_reextract=False
    ):
        
        print(f"Extracting frames from video: {video_name}")
        
        os.makedirs(output_folder, exist_ok=True)

        video_path = os.path.join(VIDEOS_DIR, participant_id, f"{video_name}.mp4")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")
        
        required_frames = self._get_required_frame_numbers(video_name)
        required_frames = sorted({int(f) for f in required_frames})
        required_frame_set = set(required_frames)

        print(f"Need to extract {len(required_frames)} annotated frames for video: {video_name}")

        if not force_reextract and self._required_frames_extracted(output_folder, required_frames):
            print(f"All required frames already extracted for video: {video_name}")
            cap.release()
            return

        frame_idx = 0
        saved_count = 0
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if frame_idx not in required_frame_set:
                    frame_idx += 1
                    continue

                filename = (
                    f"{frame_idx}.png"
                )

                save_path = os.path.join(output_folder, filename)
                # check if file already exists to avoid redundant processing
                if os.path.exists(save_path):
                    print(f"Frame {frame_idx} already exists at {save_path}, skipping.")
                    frame_idx += 1
                    continue

                if resize_to:
                    frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)

                if not dry_run:
                    f = frame.copy()
                    future = executor.submit(cv2.imwrite, save_path, f)
                    future.add_done_callback(lambda fut, idx=frame_idx, sp=save_path: print(f"Saved frame {idx} to {sp}"))

                saved_count += 1
                frame_idx += 1

                print(f"Processed frame {frame_idx}", end="\r")

        cap.release()

        print(f"Finished extracting frames for video: {video_name}")
        print(f"Saved {saved_count} new frames to: {output_folder}")

    def _required_frames_extracted(self, output_folder, required_frames: List[int]):
        if not required_frames:
            return False

        missing = [frame for frame in required_frames if not os.path.exists(os.path.join(output_folder, f"{frame}.png"))]
        if missing:
            print(f"Missing {len(missing)} required frames in {output_folder}")
            return False

        print(f"Found all {len(required_frames)} required frames in {output_folder}")
        return True

    def _all_frames_extracted(self, output_folder, total_frames):
        if total_frames <= 0:
            return False

        existing_frames = [f for f in os.listdir(output_folder) if f.endswith(".png")]
        
        print(f"Found {len(existing_frames)} existing frames in {output_folder}, total frames in video: {total_frames}")
        return len(existing_frames) >= total_frames    
        
            
    def extract_video_masks(self, video_name, video_frames_dir, masks_output_dir, dry_run=False):
        print(f"Extracting masks for video: {video_name}")

        video_associations = self.assocs_annotations[video_name]
        video_masks = self.masks_annotations[video_name]
        video_attached_masks_dir = os.path.join(self.attached_masks_dir, video_name)
        video_unattached_masks_dir = os.path.join(self.unattached_masks_dir, video_name)

        with ThreadPoolExecutor(max_workers=8) as executor:
            for association_id, association_details in video_associations.items():
                association_name = association_details['name']
                
                if "Track" in association_name or "skipped" in association_name.lower():
                    print(f"Skipping association {association_id} with name {association_name} as it contains 'Track' or 'skipped'.")
                    continue

                print(f"Processing association ID: {association_id}, {association_name}")

                association_output_dir = os.path.join(masks_output_dir, association_name)
                os.makedirs(association_output_dir, exist_ok=True)

                for track in association_details["tracks"]:
                    if not track["masks"]:
                        continue

                    for mask_id in track["masks"]:
                        mask = video_masks[mask_id]
                        frame_num = mask["frame_number"]
                        frame_path = os.path.join(video_frames_dir, f"{frame_num}.png")

                        if not os.path.exists(frame_path):
                            print(f"Frame {frame_num} does not exist at {frame_path}")
                            continue

                        alpha = self._load_alpha(mask_id, video_attached_masks_dir, video_unattached_masks_dir)
                        if alpha is None:
                            print(f"No mask found for mask ID {mask_id} at frame {frame_num}")
                            continue

                        if not dry_run:
                            save_path = os.path.join(association_output_dir, f"{frame_num}.png")
                            if os.path.exists(save_path):
                                print(f"Mask for frame {frame_num} already exists at {save_path}, skipping.")
                                continue
                            a = alpha.copy()
                            future = executor.submit(self._save_rgba_alpha, a, save_path, (a.shape[1], a.shape[0]))
                            future.add_done_callback(
                                lambda fut, mid=mask_id, fn=frame_num, sp=save_path:
                                    print(f"Saved mask {mid} for frame {fn} to {sp}")
                            )

    def _load_alpha(self, mask_id, video_attached_masks_dir, video_unattached_masks_dir):
        for mask_dir in (video_attached_masks_dir, video_unattached_masks_dir):
            mask_path = os.path.join(mask_dir, f"{mask_id}.png")
            if os.path.exists(mask_path):
                arr = np.array(Image.open(mask_path))
                if arr.ndim == 3 and arr.shape[2] == 4:
                    arr = arr[..., 3]
                elif arr.ndim == 3:
                    arr = np.any(arr[..., :3] > 0, axis=2).astype(np.uint8) * 255
                return (arr > 0).astype(np.uint8) * 255
        return None

    def _save_rgba_alpha(self, alpha, output_path, size):
        rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        rgba[..., 3] = alpha
        Image.fromarray(rgba, mode="RGBA").save(output_path)

if __name__ == "__main__":

    # for testing
    DRY_RUN = False
    FORCE_REEXTRACT = False
    PARTICIPANT_ID = "P01"
    VIDEO_NAME = "P01-20240202-110250"

    pipeline = HDEPICExtractionPipeline(MASKS_ANNOTATIONS_JSON_PATH, ASSOC_ANNOTATIONS_JSON_PATH, ATTACHED_MASKS_DIR, UNATTACHED_MASKS_DIR)

    output_dir = os.path.join(EXTRACTED_FRAMES_AND_MASKS_ROOT_FOLDER, VIDEO_NAME)
    frames_output_dir = os.path.join(output_dir, "images")
    masks_output_dir = os.path.join(output_dir)
    
    pipeline.extract_frames(PARTICIPANT_ID, VIDEO_NAME, frames_output_dir, dry_run=DRY_RUN, force_reextract=FORCE_REEXTRACT)
    pipeline.extract_video_masks(VIDEO_NAME, frames_output_dir, masks_output_dir, dry_run=DRY_RUN)

