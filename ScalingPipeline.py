import json
import os
import math
import numpy as np
import pandas as pd
from pathlib import Path


import bpy
import cv2
from scipy.spatial.transform import Rotation


class ScalingPipeline:
    def __init__(self, masks_annotation_json_path, 
                 assoc_annotation_json_path,
                 attached_masks_dir, 
                 unattached_masks_dir, 
                 slam_and_gaze_root_dir,
                 videos_root_dir,
                 recon_input_root_dir=Path(__file__).parent / "visualization",
                 scaled_output_root_dir=Path(__file__).parent / "visualization_scaled"):
        
        self.masks_annotation_json_path = masks_annotation_json_path
        self.assoc_annotation_json_path = assoc_annotation_json_path
        self.attached_masks_dir = attached_masks_dir
        self.unattached_masks_dir = unattached_masks_dir
        self.slam_and_gaze_root_dir = slam_and_gaze_root_dir
        self.videos_root_dir = videos_root_dir
        self.recon_input_root_dir = recon_input_root_dir
        self.scaled_output_root_dir = scaled_output_root_dir
        
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

    def _load_calibration(self, calib_jsonl):
        with open(calib_jsonl) as f:
            calib = json.loads(f.readline())
        rgb = next(c for c in calib["CameraCalibrations"] if c["Label"] == "camera-rgb")
        params = rgb["Projection"]["Params"]
        f_rgb  = params[0]
        cx_rgb = params[1]
        cy_rgb = params[2]
        t_dc = rgb["T_Device_Camera"]["Translation"]
        q_dc = rgb["T_Device_Camera"]["UnitQuaternion"]
        T_device_cam_t = np.array(t_dc)
        T_device_cam_q = np.array([q_dc[1][0], q_dc[1][1], q_dc[1][2], q_dc[0]])  # xyzw
        return f_rgb, cx_rgb, cy_rgb, T_device_cam_t, T_device_cam_q

    def _get_mask_path(self, video_name, mask_id):
        attached = Path(self.attached_masks_dir) / video_name /f"{mask_id}.png"
        if attached.exists():
            return attached
        unattached = Path(self.unattached_masks_dir) / video_name /f"{mask_id}.png"
        if unattached.exists():
            return unattached
        return None
    
    def _get_scale_for_association(self, assoc_name, video_id,
                                traj, mp4_to_vrs, focal_length, T_device_cam_t, R_device_cam):
        
        video_assocs = self.assocs_annotations.get(video_id, {})
        video_masks  = self.masks_annotations.get(video_id, {})
        candidates   = []

        for assoc_id, assoc in video_assocs.items():
            if assoc["name"] != assoc_name:
                continue
            for track in assoc.get("tracks", []):
                for mask_id in track.get("masks", []):
                    entry = video_masks.get(mask_id)
                    if not entry or not entry.get("3d_location"):
                        continue
                    assoc_3d_pos = entry["3d_location"]
                    frame_idx = entry.get("frame_number")
                    if frame_idx is None:
                        continue
                    try:
                        camera_pose = self._get_camera_pose_for_frame(frame_idx, traj, mp4_to_vrs)
                        z = self._get_metric_z(assoc_3d_pos, camera_pose, T_device_cam_t, R_device_cam) # get metric z (depth) of the object from camera
                        if z <= 0: # invalid depth (behind camera dont make sense)s
                            continue
                        real_size = None
                        mask_pixel_length = self._get_mask_pixel_length(video_id, mask_id)
                        real_size, _ = self._get_real_world_size(
                            assoc_3d_pos, camera_pose, mask_pixel_length, focal_length, T_device_cam_t, R_device_cam
                        )
                        candidates.append((assoc_3d_pos, z, real_size))
                        
                    except Exception as e:
                        print(f"  [WARN] mask {mask_id} frame {frame_idx}: {e}")
                        continue

        if not candidates:
            return None, None, None

        # take the candidate whose depth is closest to the median depth of all candidates, to be robust to outliers 
        # (e.g. if some masks are very inaccurate and give incorrect depths/sizes, we dont want to choose those as our scale reference)
        zs       = [z for _, z, _ in candidates]
        median_z = float(np.median(zs))
        best     = min(candidates, key=lambda x: abs(x[1] - median_z))
        
        return best[2], best[1], best[0]  # real_size, metric_z, pos

    def _get_camera_pose_for_frame(self, frame_idx, traj, mp4_to_vrs):
        frame_idx = int(frame_idx)
        vrs_ns = mp4_to_vrs.iloc[frame_idx]["vrs_device_time_ns"] # check the vrs timestamp for this frame
        idx = np.argmin(np.abs(traj["tracking_timestamp_ns"] - vrs_ns)) # get traj datapoint with closest timestamp to the vrs timestamp for this frame
        return traj.iloc[int(idx)]
    
    def _get_metric_z(self, pos_world, camera_pose, T_device_cam_t, R_device_cam):
        obj_world = np.array(pos_world) # object position in world coordinates from annotation
        R, t = self._pose_to_R_t_rgb(camera_pose, T_device_cam_t, R_device_cam)
        obj_cam = R @ obj_world + t # object position in camera coordinates. the z value gives us the "depth" of the object from the camera
        return float(obj_cam[2])
    
    def _pose_to_R_t_rgb(self,camera_pose, T_device_cam_t, R_device_cam):
        # Convert from world-device to world-camera coordinates
        t_wd = np.array([
            camera_pose["tx_world_device"], 
            camera_pose["ty_world_device"],
            camera_pose["tz_world_device"]
        ])
        q_wd = np.array([
            camera_pose["qx_world_device"],
            camera_pose["qy_world_device"],
            camera_pose["qz_world_device"],
            camera_pose["qw_world_device"]
        ])
        R_wd = Rotation.from_quat(q_wd).as_matrix()
        R_wc = R_wd @ R_device_cam # rotation from world to camera
        t_wc = R_wd @ T_device_cam_t + t_wd # translation from world to camera
        R_cam_world = R_wc.T
        t_cam_world = -R_cam_world @ t_wc
        return R_cam_world, t_cam_world
    
    def _get_mask_pixel_length(self, video_name, mask_id):

        mask_path = self._get_mask_path(video_name, mask_id)
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # TODO: switch to finding maxferet diameter instead of bbox long, since object is not necessarily aligned with camera axes
        rows_any = np.any(mask_img > 0, axis=1)
        cols_any = np.any(mask_img > 0, axis=0)
        h = int(np.sum(rows_any))
        w = int(np.sum(cols_any))

        # TODO: make it robust to the perspective/angle that we view objects from
        # e.g. if we look at a fork from the end straight-on, the bbox is tiny, causing the scale to be tiny, 
        # since we do not see the actual full length of the fork
        bbox_long = max(h, w) # get longest edge of the bounding box as a single "size" metric for the object
        
        return bbox_long
    
    def _get_real_world_size(self, assoc_pos_world, camera_pose, mask_pixel_length, focal_length, T_device_cam_t, R_device_cam):
        
        # use pinhole camera model to get real world size from pixel size and depth
        assoc_pos_world = np.array(assoc_pos_world)
        R, t = self._pose_to_R_t_rgb(camera_pose, T_device_cam_t, R_device_cam)
        assoc_pos_cam = R @ assoc_pos_world + t # position of object/association in camera coordinates
        cam_to_assoc_dist = float(np.linalg.norm(assoc_pos_cam)) # get euclidean distance from camera to object
        real_size = (float(mask_pixel_length) / focal_length) * cam_to_assoc_dist # y = x * z / f  (similar triangles in pinhole camera model, where x is pixel length, y is real world length, z is depth from camera, f is focal length)
        return real_size, cam_to_assoc_dist

    def _get_video_num(self, video_name):
        participant_id = video_name.split('-')[0]
        vrs_to_multi_slam_json_path = Path(self.slam_and_gaze_root_dir) / participant_id / "SLAM" / "multi" / "vrs_to_multi_slam.json"
        
        with open(vrs_to_multi_slam_json_path, "r") as f:
            vrs_to_multi_slam = json.load(f)
            video_num = vrs_to_multi_slam.get(f"{participant_id}/{video_name}.vrs")

        return video_num

    def _load_trajectory(self, traj_csv, mp4_vrs_csv):
        traj = pd.read_csv(traj_csv)
        traj["tracking_timestamp_ns"] = traj["tracking_timestamp_us"] * 1000
        mp4_to_vrs = pd.read_csv(mp4_vrs_csv)
        return traj, mp4_to_vrs

    def _iter_result_glbs(self, root):
        for obj_dir in sorted(Path(root).iterdir()):
            if not obj_dir.is_dir():
                continue
            for nested in obj_dir.iterdir():
                if not nested.is_dir():
                    continue
                glb = nested / "result.glb"
                if not glb.exists():
                    continue
                parts = nested.name.split("_")
                try:
                    mv_idx = next(i for i, p in enumerate(parts) if p == "mv")
                    assoc_name = " ".join(parts[1:mv_idx])
                except StopIteration:
                    assoc_name = obj_dir.name
                yield assoc_name, str(glb)
    
    @staticmethod
    def import_glb(glb_path, location, name):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=glb_path)
        new_objs = set(bpy.data.objects) - before
        roots = [o for o in new_objs if o.parent is None]
        for root in roots:
            root.location = location
            root.name = name
        return new_objs

    def _get_scaled_output_glb_path(self, glb_path):
        relative_glb_path = Path(glb_path).relative_to(self.recon_input_root_dir)
        output_glb_path = Path(self.scaled_output_root_dir) / relative_glb_path
        output_glb_path.parent.mkdir(parents=True, exist_ok=True)
        return output_glb_path

    @staticmethod
    def export_glb(output_path, objects):
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)

        # Bake current object transforms into mesh data so downstream consumers
        # don't depend on node-level scale transforms being preserved.
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        bpy.ops.export_scene.gltf(
            filepath=str(output_path),
            export_format="GLB",
            use_selection=True,
            export_apply=True,
        )

    @staticmethod
    def remove_objects(objects):
        for obj in list(objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    
    @staticmethod
    def get_mesh_max_dim(objects):
        all_verts = []
        for obj in objects:
            if obj.type != "MESH":
                continue
            for v in obj.data.vertices:
                wco = obj.matrix_world @ v.co
                all_verts.append([wco.x, wco.y, wco.z])
        if not all_verts:
            return None
        verts = np.array(all_verts)
        return float((verts.max(axis=0) - verts.min(axis=0)).max())

    @staticmethod
    def zero_root_locations(objects):
        roots = [o for o in objects if o.parent is None]
        for root in roots:
            root.location = (0.0, 0.0, 0.0)


    def _copy_debug_masked_images(self, video_name):
        # copy the debug masked images from original recon input dir to the scaled output dir, so that we can visualize the masks alongside the scaled GLBs
        video_dir = Path(self.recon_input_root_dir) / video_name
        if not video_dir.is_dir():
            return
        for assoc_root_dir in video_dir.iterdir():
            print(f"Checking association directory: {assoc_root_dir}")
            if not assoc_root_dir.is_dir():
                continue
            
            for assoc_dir in assoc_root_dir.iterdir():
                if not assoc_dir.is_dir():
                    continue
                debug_masked_images_dir = assoc_dir / "debug_masked_images"
                if not debug_masked_images_dir.exists():
                    continue
                output_debug_masked_images_dir = Path(self.scaled_output_root_dir) / video_dir.name / assoc_root_dir.name / assoc_dir.name / "debug_masked_images"
                output_debug_masked_images_dir.mkdir(parents=True, exist_ok=True)
                for img_file in debug_masked_images_dir.iterdir():
                    print(f"Copying debug masked image: {img_file} to {output_debug_masked_images_dir}")
                    if img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        output_img_file = output_debug_masked_images_dir / img_file.name
                        with open(img_file, "rb") as src, open(output_img_file, "wb") as dst:
                            dst.write(src.read())

    def run_pipeline(self, video_name, dry_run=False):
        
        participant_id = video_name.split('-')[0]
        video_num = self._get_video_num(video_name)
        skipped = 0

        video_traj_path = Path(self.slam_and_gaze_root_dir) / participant_id / "SLAM" / "multi" / video_num / video_num / "slam" / "closed_loop_trajectory.csv"
        calib_jsonl_path = Path(self.slam_and_gaze_root_dir) / participant_id / "SLAM" / "multi" / video_num / video_num / "slam" / "online_calibration.jsonl"
        mp4_to_vrs_csv_path = Path(self.videos_root_dir) / participant_id / f"{video_name}_mp4_to_vrs_time_ns.csv"

        traj, mp4_to_vrs = self._load_trajectory(video_traj_path, mp4_to_vrs_csv_path)
        
        # fx = focal length in pixels, cx_rgb = principal point x coordinate, cy_rgb = principal point y coordinate, 
        # T_device_cam_t = translation vector from device to camera, T_device_cam_q = quaternion rotation from device to camera
        fx, cx_rgb, cy_rgb, T_device_cam_t, T_device_cam_q = self._load_calibration(calib_jsonl_path)
        R_device_cam = Rotation.from_quat(T_device_cam_q).as_matrix()
        print(f"Calibration: f={fx:.1f}  cx={cx_rgb:.1f}  cy={cy_rgb:.1f}")

        reconstructed_objects_glb_list = list(self._iter_result_glbs(Path(self.recon_input_root_dir) / video_name))

        for assoc_name, glb_path in reconstructed_objects_glb_list:
            print(f"Processing association: {assoc_name}")
       
            real_size, metric_z, pos = self._get_scale_for_association(
                assoc_name, video_name,
                traj, mp4_to_vrs, fx, T_device_cam_t, R_device_cam
            )

            if pos is None:
                print(f"[SKIP] no valid pose/depth found for '{assoc_name}'")
                skipped += 1
                continue

            print(f"Found scale reference for '{assoc_name}': real_size={real_size:.3f}m, metric_z={metric_z:.3f}m, pos={pos}")

            new_objs = self.import_glb(glb_path, tuple(pos), assoc_name)
            bpy.context.view_layer.update()

            scale_factor = None

            if real_size is not None:
                mesh_dim = self.get_mesh_max_dim(new_objs)
                if mesh_dim and mesh_dim > 0:
                    scale_factor = real_size / mesh_dim
                    print(f"real_size={real_size:.3f}m  mesh_dim={mesh_dim:.4f}  scale={scale_factor:.4f}")
                else:
                    print(f"real_size={real_size:.3f}m  mesh_dim=BAD")
            elif metric_z is not None:
                all_z = []
                for obj in new_objs:
                    if obj.type != "MESH":
                        continue
                    for v in obj.data.vertices:
                        all_z.append(abs((obj.matrix_world @ v.co).z))
                if all_z:
                    mvs_z = float(np.mean(all_z))
                    scale_factor = metric_z / mvs_z if mvs_z > 0 else None
                    if scale_factor is not None:
                        print(f"z-fallback: metric_z={metric_z:.3f}m  mvs_z={mvs_z:.4f}  scale={scale_factor:.4f}")
                    else:
                        print(f"z-fallback: metric_z={metric_z:.3f}m  mvs_z={mvs_z:.4f}  scale=BAD")
                else:
                    print("z-fallback: no vertices")
            else:
                print("no metric depth")

            if scale_factor is None:
                print(f"[SKIP] no valid scale factor found for '{assoc_name}'")
                skipped += 1
                self.remove_objects(new_objs)
                continue

            roots = [o for o in new_objs if o.parent is None]
            for root in roots:
                root.scale = (scale_factor, scale_factor, scale_factor)

            # Ensure exported GLBs have zeroed XYZ at object/root level.
            self.zero_root_locations(new_objs)

            output_glb_path = self._get_scaled_output_glb_path(glb_path)
            self.export_glb(output_glb_path, new_objs)
            print(f"Saved scaled GLB to {output_glb_path}")

            self.remove_objects(new_objs)

        print(f"Finished scaling for {video_name}: skipped={skipped}")

        # Copy debug masked images to the scaled output directory
        self._copy_debug_masked_images(video_name)
        print(f"Copied debug masked images for {video_name} to scaled output directory")

if __name__ == "__main__":
    
    # Example usage
    pipeline = ScalingPipeline(
        masks_annotation_json_path="/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/hd-epic-annotations/scene-and-object-movements/mask_info.json",
        assoc_annotation_json_path="/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/hd-epic-annotations/scene-and-object-movements/assoc_info.json",
        attached_masks_dir="/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/masks/masks",
        unattached_masks_dir="/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/masks/unattached_masks",
        slam_and_gaze_root_dir="/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/HD-EPIC/SLAM-and-Gaze",
        videos_root_dir="/home/ghdl2/rds/hpc-work/MPhil Dissertation/data/hd-epic/HD-EPIC/Videos"
    )

    pipeline.run_pipeline("P01-20240202-110250", dry_run=False)