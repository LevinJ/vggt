"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Fri Jul 04 2025
*  File : test_vggt.py
******************************************* -->

"""
import numpy as np
import open3d as o3d
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import os


class TestVGGT(object):
    def __init__(self):
        return

    def run(self):
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model = VGGT()
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

        predictions = self.process_images(
            "/home/levin/workspace/nerf/forward/vggt/examples/room/images", model
        )
        torch.cuda.empty_cache()

        # Now you can use the predictions for further processing or visualization
        point_cloud = self.predictions_to_open3d_pointcloud(predictions, conf_thres=5, prediction_mode="depth_map")
        o3d.visualization.draw_geometries([point_cloud], window_name="Open3D Point Cloud")

        
        return

    def process_images(self, image_folder, model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        model.eval()
        model = model.to(device)

        image_names = [
            os.path.join(image_folder, fname)
            for fname in os.listdir(image_folder)
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]
        image_names = sorted(image_names)
        print(f"Collected {len(image_names)} images.")
        images = load_and_preprocess_images(image_names).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)

        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        print("Computing world points from depth map...")
        depth_map = predictions["depth"]
        world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        predictions["world_points_from_depth"] = world_points

        return predictions

    def predictions_to_open3d_pointcloud(
        self,
        predictions,
        conf_thres=50.0,
        filter_by_frames="all",
        mask_black_bg=False,
        mask_white_bg=False,
        show_cam=True,
        mask_sky=False,
        prediction_mode="Predicted Pointmap",
    ):
        """
        Converts VGGT predictions to an Open3D point cloud.

        Args:
            predictions (dict): Dictionary containing model predictions with keys:
                - world_points: 3D point coordinates (S, H, W, 3)
                - world_points_conf: Confidence scores (S, H, W)
                - images: Input images (S, H, W, 3)
                - extrinsic: Camera extrinsic matrices (S, 3, 4)
            conf_thres (float): Percentage of low-confidence points to filter out (default: 50.0)
            filter_by_frames (str): Frame filter specification (default: "all")
            mask_black_bg (bool): Mask out black background pixels (default: False)
            mask_white_bg (bool): Mask out white background pixels (default: False)
            show_cam (bool): Include camera visualization (default: True)
            mask_sky (bool): Apply sky segmentation mask (default: False)
            prediction_mode (str): Prediction mode selector (default: "Predicted Pointmap")

        Returns:
            open3d.geometry.PointCloud: Processed 3D point cloud

        Raises:
            ValueError: If input predictions structure is invalid
        """
        if not isinstance(predictions, dict):
            raise ValueError("predictions must be a dictionary")

        if conf_thres is None:
            conf_thres = 10.0

        selected_frame_idx = None
        if filter_by_frames != "all" and filter_by_frames != "All":
            try:
                selected_frame_idx = int(filter_by_frames.split(":")[0])
            except (ValueError, IndexError):
                pass

        if "Pointmap" in prediction_mode:
            if "world_points" in predictions:
                pred_world_points = predictions["world_points"]
                pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
            else:
                pred_world_points = predictions["world_points_from_depth"]
                pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

        images = predictions["images"]
        camera_matrices = predictions["extrinsic"]

        if selected_frame_idx is not None:
            pred_world_points = pred_world_points[selected_frame_idx][None]
            pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
            images = images[selected_frame_idx][None]
            camera_matrices = camera_matrices[selected_frame_idx][None]

        vertices_3d = pred_world_points.reshape(-1, 3)
        if images.ndim == 4 and images.shape[1] == 3:
            colors_rgb = np.transpose(images, (0, 2, 3, 1))
        else:
            colors_rgb = images
        colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

        conf = pred_world_points_conf.reshape(-1)
        if conf_thres == 0.0:
            conf_threshold = 0.0
        else:
            conf_threshold = np.percentile(conf, conf_thres)

        conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

        if mask_black_bg:
            black_bg_mask = colors_rgb.sum(axis=1) >= 16
            conf_mask = conf_mask & black_bg_mask

        if mask_white_bg:
            white_bg_mask = ~((colors_rgb[:, 0] > 240) & (colors_rgb[:, 1] > 240) & (colors_rgb[:, 2] > 240))
            conf_mask = conf_mask & white_bg_mask

        vertices_3d = vertices_3d[conf_mask]
        colors_rgb = colors_rgb[conf_mask]

        if vertices_3d is None or np.asarray(vertices_3d).size == 0:
            vertices_3d = np.array([[1, 0, 0]])
            colors_rgb = np.array([[255, 255, 255]])

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(vertices_3d)
        point_cloud.colors = o3d.utility.Vector3dVector(colors_rgb / 255.0)

        return point_cloud


if __name__ == "__main__":
    obj = TestVGGT()
    obj.run()
