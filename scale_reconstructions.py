from argparse import ArgumentParser
from datetime import datetime
import json
import os
import time
from pathlib import Path

from ScalingPipeline import ScalingPipeline

def _default_report_path():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).parent / "logs" / f"scale_report_{timestamp}.json"


def main():
    parser = ArgumentParser(description="Run scaling pipeline over all visualization videos and write a report.")
    parser.add_argument(
        "--report-path",
        default=str(_default_report_path()),
        help="Path for JSON summary report output.",
    )
    args = parser.parse_args()

    pipeline = ScalingPipeline(
        masks_annotation_json_path="/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/hd-epic-annotations/scene-and-object-movements/mask_info.json",
        assoc_annotation_json_path="/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/hd-epic-annotations/scene-and-object-movements/assoc_info.json",
        attached_masks_dir="/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/masks/masks",
        unattached_masks_dir="/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/masks/unattached_masks",
        slam_and_gaze_root_dir="/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/HD-EPIC/SLAM-and-Gaze",
        videos_root_dir="/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/HD-EPIC/Videos",
    )

    visualisations_root_dir = Path("/home/ghdl2/rds/hpc-work/MPhil Dissertation/pipeline/visualization")
    report = {
        "started_at": datetime.now().isoformat(),
        "videos": [],
        "totals": {
            "videos_attempted": 0,
            "videos_failed": 0,
            "total_candidates": 0,
            "imported": 0,
            "skipped": 0,
        },
    }

    for video_name in sorted(os.listdir(visualisations_root_dir)):
        
        # check if video already processed
        scaled_visualisations_root_dir = Path("/home/ghdl2/rds/hpc-work/MPhil Dissertation/pipeline/visualization_scaled")
        video_output_dir = scaled_visualisations_root_dir / video_name
        if video_output_dir.exists() and any(video_output_dir.iterdir()):
            print(f"Skipping {video_name} as it has already been processed.")
            continue
        
        print(f"Running pipeline for {video_name}...")
        start_time = time.time()
        report["totals"]["videos_attempted"] += 1

        try:
            video_report = pipeline.run_pipeline(video_name, dry_run=False)
            elapsed = time.time() - start_time
            video_report["elapsed_seconds"] = round(elapsed, 2)
            video_report["status"] = "ok"
            report["videos"].append(video_report)

            report["totals"]["total_candidates"] += video_report["total_candidates"]
            report["totals"]["imported"] += video_report["imported"]
            report["totals"]["skipped"] += video_report["skipped"]
        except Exception as exc:
            elapsed = time.time() - start_time
            report["totals"]["videos_failed"] += 1
            report["videos"].append(
                {
                    "video_name": video_name,
                    "status": "failed",
                    "elapsed_seconds": round(elapsed, 2),
                    "error": str(exc),
                }
            )
            print(f"[ERROR] Pipeline failed for {video_name}: {exc}")

        print(f"Finished pipeline for {video_name} in {time.time() - start_time:.2f} seconds")

    report["finished_at"] = datetime.now().isoformat()
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved scale report to {report_path}")


if __name__ == "__main__":
    main()