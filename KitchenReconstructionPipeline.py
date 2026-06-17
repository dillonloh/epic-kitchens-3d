import json
import colorsys
from pathlib import Path

import bpy
import numpy as np


class KitchenReconstructionPipeline:
    """Populate a participant kitchen blend with already-scaled GLBs at annotated 3D positions."""

    def __init__(
        self,
        object_glb_root_dir=Path(__file__).parent / "visualization_scaled",
        masks_annotation_json_path=Path(__file__).resolve().parents[1]
        / "data"
        / "hd-epic"
        / "hd-epic-annotations"
        / "scene-and-object-movements"
        / "mask_info.json",
        assoc_annotation_json_path=Path(__file__).resolve().parents[1]
        / "data"
        / "hd-epic"
        / "hd-epic-annotations"
        / "scene-and-object-movements"
        / "assoc_info.json",
        kitchen_blend_root_dir=Path(__file__).resolve().parents[1] / "data" / "hd-epic" / "HD-EPIC" / "Digital-Twin",
        output_root_dir=Path(__file__).parent / "visualization_scene",
        enable_colours=True,
    ):
        self.object_glb_root_dir = Path(object_glb_root_dir)
        self.masks_annotation_json_path = Path(masks_annotation_json_path)
        self.assoc_annotation_json_path = Path(assoc_annotation_json_path)
        self.kitchen_blend_root_dir = Path(kitchen_blend_root_dir)
        self.output_root_dir = Path(output_root_dir)
        self.enable_colours = bool(enable_colours)

        with open(self.masks_annotation_json_path, "r") as f:
            self.masks_annotations = json.load(f)
        with open(self.assoc_annotation_json_path, "r") as f:
            self.assocs_annotations = json.load(f)

    def _get_world_position_for_association(self, assoc_name, video_name):
        video_assocs = self.assocs_annotations.get(video_name, {})
        video_masks = self.masks_annotations.get(video_name, {})
        positions = []

        for _, assoc in video_assocs.items():
            if assoc.get("name") != assoc_name:
                continue

            for track in assoc.get("tracks", []):
                for mask_id in track.get("masks", []):
                    entry = video_masks.get(mask_id)
                    if not entry or not entry.get("3d_location"):
                        continue

                    pos = entry["3d_location"]
                    if len(pos) != 3:
                        continue
                    positions.append(np.array(pos, dtype=float))

        if not positions:
            return None

        # Use component-wise median across mask positions for robustness to outliers.
        stacked = np.vstack(positions)
        return np.median(stacked, axis=0).tolist()

    @staticmethod
    def _iter_result_glbs(video_glb_root):
        for assoc_dir in sorted(Path(video_glb_root).iterdir()):
            if not assoc_dir.is_dir():
                continue

            for run_dir in sorted(assoc_dir.iterdir()):
                if not run_dir.is_dir():
                    continue

                result_glb = run_dir / "result.glb"
                if not result_glb.exists():
                    continue

                yield assoc_dir.name, result_glb

    @staticmethod
    def _import_glb(glb_path, assoc_name, location):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(glb_path))
        new_objs = set(bpy.data.objects) - before

        for root in [o for o in new_objs if o.parent is None]:
            root.name = assoc_name
            root.location = location

        return new_objs

    @staticmethod
    def _get_colours(n):
        return [colorsys.hsv_to_rgb(i / max(n, 1), 0.8, 0.9) for i in range(n)]

    @staticmethod
    def _assign_colour(objs, colour, material_name="mat"):
        mat = bpy.data.materials.new(name=material_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*colour, 1.0)

        for obj in objs:
            if obj.type == "MESH":
                obj.data.materials.clear()
                obj.data.materials.append(mat)

    def run_pipeline(self, video_name, output_path=None):
        
        participant_id = video_name.split("-")[0]
        kitchen_blend_path = self.kitchen_blend_root_dir / f"{participant_id}_final.blend"
        video_glb_root = self.object_glb_root_dir / video_name

        if not kitchen_blend_path.exists():
            raise FileNotFoundError(f"Kitchen blend not found: {kitchen_blend_path}")
        if not video_glb_root.exists():
            raise FileNotFoundError(f"Object GLB directory not found: {video_glb_root}")

        glbs = list(self._iter_result_glbs(video_glb_root))
        if not glbs:
            raise RuntimeError(f"No result.glb files found under: {video_glb_root}")

        colours = self._get_colours(len(glbs)) if self.enable_colours else []

        bpy.ops.wm.open_mainfile(filepath=str(kitchen_blend_path))

        imported_count = 0
        skipped_count = 0
        for idx, (assoc_name, glb_path) in enumerate(glbs):
            pos = self._get_world_position_for_association(assoc_name, video_name)
            if pos is None:
                print(f"[SKIP] no annotated 3D position found for '{assoc_name}'")
                skipped_count += 1
                continue

            imported_objs = self._import_glb(glb_path, assoc_name, tuple(pos))
            if self.enable_colours:
                self._assign_colour(
                    imported_objs,
                    colours[idx],
                    material_name=f"mat_{assoc_name}_{idx}",
                )
            imported_count += 1
            if self.enable_colours:
                print(f"[IMPORT] {assoc_name}: {glb_path} at {pos} with colour {colours[idx]}")
            else:
                print(f"[IMPORT] {assoc_name}: {glb_path} at {pos}")

        bpy.context.view_layer.update()

        if output_path is None:
            self.output_root_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_root_dir / f"{video_name}_kitchen_populated.blend"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        bpy.ops.wm.save_mainfile(filepath=str(output_path))
        print(f"Saved populated scene: {output_path} ({imported_count} imported, {skipped_count} skipped)")

        return output_path


if __name__ == "__main__":
    pipeline = KitchenReconstructionPipeline()
    pipeline.run_pipeline("P01-20240202-110250")




