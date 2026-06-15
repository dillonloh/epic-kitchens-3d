import json
import colorsys
from pathlib import Path

import bpy
import numpy as np


class GridReconstructionPipeline:
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
    def _iter_result_glbs_with_grid(video_glb_root):
        glbs = list(GridReconstructionPipeline._iter_result_glbs(video_glb_root))
        if not glbs:
            return []

        grid_cols = max(1, int(np.ceil(np.sqrt(len(glbs)))))
        grid_glbs = []
        for idx, (assoc_name, glb_path) in enumerate(glbs):
            row = idx // grid_cols
            col = idx % grid_cols
            grid_glbs.append((assoc_name, glb_path, row, col))

        return grid_glbs

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

    def _prepare_output_path(self, output_path, default_filename):
        if output_path is None:
            self.output_root_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_root_dir / default_filename
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        return output_path

    def run_grid_pipeline(self, video_name, output_path=None, grid_spacing=0.6):
        video_glb_root = self.object_glb_root_dir / video_name

        if not video_glb_root.exists():
            raise FileNotFoundError(f"Object GLB directory not found: {video_glb_root}")

        grid_glbs = self._iter_result_glbs_with_grid(video_glb_root)
        if not grid_glbs:
            raise RuntimeError(f"No result.glb files found under: {video_glb_root}")

        colours = self._get_colours(len(grid_glbs)) if self.enable_colours else []

        bpy.ops.wm.read_factory_settings(use_empty=True)

        imported_count = 0
        for idx, (assoc_name, glb_path, row, col) in enumerate(grid_glbs):
            grid_pos = (col * grid_spacing, -row * grid_spacing, 0.0)
            imported_objs = self._import_glb(glb_path, assoc_name, grid_pos)
            if self.enable_colours:
                self._assign_colour(
                    imported_objs,
                    colours[idx],
                    material_name=f"mat_{assoc_name}_{idx}",
                )
            imported_count += 1
            if self.enable_colours:
                print(f"[GRID] {assoc_name}: {glb_path} at {grid_pos} with colour {colours[idx]}")
            else:
                print(f"[GRID] {assoc_name}: {glb_path} at {grid_pos}")

        bpy.context.view_layer.update()

        output_path = self._prepare_output_path(output_path, f"{video_name}_grid.blend")

        bpy.ops.wm.save_mainfile(filepath=str(output_path))
        print(f"Saved grid scene: {output_path} ({imported_count} imported)")

        return output_path

    def run_pipeline(self, video_name, output_path=None, grid_spacing=0.6):
        return self.run_grid_pipeline(
            video_name=video_name,
            output_path=output_path,
            grid_spacing=grid_spacing,
        )


if __name__ == "__main__":
    pipeline = GridReconstructionPipeline()
    pipeline.run_grid_pipeline("P01-20240202-110250")




