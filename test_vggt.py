"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Fri Jul 04 2025
*  File : test_vggt.py
******************************************* -->

"""
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import os
import open3d as o3d


class App(object):
    def __init__(self):
        return

    def run(self):
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model = VGGT()
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

        predictions = self.process_images(
            "/home/levin/workspace/nerf/forward/vggt/examples/room/images", model
        )

        # Now you can use the predictions for further processing or visualization

        # Display world points using open3d
        print("Displaying world points using open3d...")
        world_points = predictions["world_points_from_depth"]

        # Reshape world points to (N*H*W, 3) before visualization
        world_points = world_points.reshape(-1, 3)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(world_points)
        o3d.visualization.draw_geometries([point_cloud])

        torch.cuda.empty_cache()
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


if __name__ == "__main__":
    obj = App()
    obj.run()
