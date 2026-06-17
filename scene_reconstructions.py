import json
import os
import time

from pathlib import Path
import bpy
import numpy as np

from KitchenReconstructionPipeline import KitchenReconstructionPipeline
from GridReconstructionPipeline import GridReconstructionPipeline
    
kitchen_pipeline = KitchenReconstructionPipeline(
        object_glb_root_dir="/home/ghdl2/rds/hpc-work/MPhil Dissertation/pipeline/visualization_scaled",
        masks_annotation_json_path="/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/hd-epic-annotations/scene-and-object-movements/mask_info.json",
        assoc_annotation_json_path="/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/hd-epic-annotations/scene-and-object-movements/assoc_info.json",
        kitchen_blend_root_dir="/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/HD-EPIC/Digital-Twin",
        output_root_dir="/home/ghdl2/rds/hpc-work/MPhil Dissertation/pipeline/visualization_scene"
)
    

grid_pipeline = GridReconstructionPipeline()

visualisations_root_dir = "/home/ghdl2/rds/hpc-work/MPhil Dissertation/pipeline/visualization_scaled"

video_names = ["P02-20240209-184316", "P05-20240424-175101"]
# for video_name in os.listdir(visualisations_root_dir):

for video_name in video_names:
    print(f"Running pipeline for {video_name}...")
    
    start_time = time.time()

    kitchen_pipeline.run_pipeline(video_name)
    grid_pipeline.run_pipeline(video_name)

    end_time = time.time()

    print(f"Finished pipeline for {video_name} in {end_time - start_time:.2f} seconds")