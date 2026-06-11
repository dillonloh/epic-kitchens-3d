import json
import os
import numpy as np
from pathlib import Path

import cv2
from scipy.spatial.transform import Rotation


class ScalingPipeline:
    def __init__(self, masks_annotation_json_path, 
                 assoc_annotation_json_path, 
                 calibration_json_path, 
                 attached_masks_dir, 
                 unattached_masks_dir, 
                 traj_root_dir,
                 recon_input_root_dir=Path(__file__).parent / "visualization",
                 scaled_output_root_dir=Path(__file__).parent / "visualization_scaled"):
        
        self.masks_annotation_json_path = masks_annotation_json_path
        self.assoc_annotation_json_path = assoc_annotation_json_path
        self.attached_masks_dir = attached_masks_dir
        self.unattached_masks_dir = unattached_masks_dir
        self.calibration_json_path = calibration_json_path
        self.traj_root_dir = traj_root_dir
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

    def _get_mask_path(self, mask_id):
        attached = Path(self.attached_masks_dir) / f"{mask_id}.png"
        if attached.exists():
            return attached
        unattached = Path(self.unattached_masks_dir) / f"{mask_id}.png"
        if unattached.exists():
            return unattached
        return None
    
    def _get_scale_for_association(self, assoc_name, assoc_info, mask_info, video_id,
                                traj, mp4_to_vrs, focal_length, T_device_cam_t, R_device_cam):
        
        video_assocs = assoc_info.get(video_id, {})
        video_masks  = mask_info.get(video_id, {})
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
                        mask_pixel_length = self._get_mask_pixel_length(mask_id)
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
    
    def _get_mask_path(self, mask_id):
        attached = Path(self.attached_masks) / f"{mask_id}.png"
        if attached.exists():
            return attached
        unattached = Path(self.unattached_masks) / f"{mask_id}.png"
        if unattached.exists():
            return unattached
        return None

    def _get_mask_pixel_length(self, mask_id):

        mask_path = self._get_mask_path(mask_id)
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

    def run_pipeline(self, video_name, dry_run=False):
        pass
