"""Pyrender-based mesh renderer that colors vertices with the mesh-to-RGB
colormaps (or a metallic material) — the renderer behind all control-image
rendering in this repo."""
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import trimesh
import pyrender
import numpy as np
import colorsys
import cv2
import torch

from posedreamer.utils.paths import WEIGHTS_DIR


class Renderer(object):

    def __init__(self, focal_length=600, principal_point=None, img_w=512, img_h=512, faces=None,
                 same_mesh_color=False, colormap="B"):
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                                   viewport_height=img_h,
                                                   point_size=1.0)
        if principal_point is not None:
            self.camera_center = [principal_point[0], principal_point[1]]
        else:
            self.camera_center = [img_w // 2, img_h // 2]
        self.focal_length = focal_length
        self.faces = faces
        self.same_mesh_color = same_mesh_color
        if colormap == "A":
            embedding = self.get_smpl_vertex_embedding()
            embedding_uint8 = (embedding * 255).astype(np.uint8)  
            colormap_bgr = cv2.applyColorMap(embedding_uint8, cv2.COLORMAP_JET)  
            colormap_rgb = colormap_bgr[:, 0, ::-1] 
            colormap_rgb = np.ascontiguousarray(colormap_rgb[:6890, ...])
        elif colormap == "B":
            colormap_rgb = np.load(str(WEIGHTS_DIR / "new_colormap_smplx.npy"))
        elif colormap == "smplx":
            colormap_rgb = np.load(str(WEIGHTS_DIR / "new_colormap_smplx.npy"))
        elif colormap == "smpl":
            colormap_rgb = np.load(str(WEIGHTS_DIR / "new_colormap_smpl.npy"))
        elif colormap == "metallic":
            colormap_rgb = None
        if colormap_rgb is not None:
            self.vertices_colors = np.ascontiguousarray(colormap_rgb)
        else:
            self.vertices_colors = None

    def get_smpl_vertex_embedding(device=torch.device("cpu")):
        # embed_url = "https://dl.fbaipublicfiles.com/densepose/data/cse/mds_d=256.npy"
        embed_path = str(WEIGHTS_DIR / "mds_d=256.npy")
        embed_map, _ = np.load(embed_path, allow_pickle=True)  
        embed_map = torch.tensor(embed_map).float()[:, 0]
        embed_map -= embed_map.min()
        embed_map /= embed_map.max()
        return embed_map.cpu().numpy()

    def render_front_view(self, verts, bg_img_rgb=None, bg_color=(0, 0, 0, 0)):
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3))
        # Create camera. Camera will always be at [0,0,0]
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                  cx=self.camera_center[0], cy=self.camera_center[1])
        camera_pose = np.eye(4)
        scene.add(camera, pose=camera_pose)

        def srgb_to_linear(img):
            img = img / 255.0
            img = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
            return np.clip(img * 255.0, 0, 255).astype(np.uint8)[:, :, :3]

        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # multiple person
        num_people = len(verts)
        # for every person in the scene
        for n in range(num_people):
            mesh = trimesh.Trimesh(verts[n], self.faces)
            mesh.apply_transform(rot)
            if self.vertices_colors is not None:
                mesh.visual.vertex_colors = self.vertices_colors
                mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False, wireframe=False)
            else:
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.1,
                    roughnessFactor=0.4,
                    alphaMode='OPAQUE',
                    emissiveFactor=(0.2, 0.2, 0.2),
                    baseColorFactor=(0.7, 0.7, 0.7, 0.8)
                )
                light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
                scene.add(light, pose=camera_pose)
                mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

            scene.add(mesh, 'mesh')

        # Alpha channel was not working previously, need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color_rgb = srgb_to_linear(color_rgba)
        if bg_img_rgb is None:
            return color_rgb
        else:
            try:
                mask = depth_map > 0
                bg_img_rgb[mask] = color_rgb[mask]
                return bg_img_rgb
            except Exception as e:
                print(f"Error while overlaying images: {color_rgba.shape} vs {depth_map.shape} vs {bg_img_rgb.shape}")
                raise e

    def render_side_view(self, verts):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        # make the centroid at the image center (the X and Y coordinates are zeros)
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0][np.newaxis, ...]  # 1*3*3
        pred_vert_arr_side = np.matmul((verts - centroid), aroundy) + centroid
        side_view = self.render_front_view(pred_vert_arr_side)
        return side_view

    def delete(self):
        """
        Need to delete before creating the renderer next time
        """
        self.renderer.delete()
