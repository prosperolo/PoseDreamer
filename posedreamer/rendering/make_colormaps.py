"""Build the PNCC-style per-vertex colormaps: normalized XYZ of a canonical
A-pose mapped to RGB. Saves new_colormap_smpl.npy and (via deformation
transfer) the color-consistent new_colormap_smplx.npy into ../weights/.
Run once after downloading the weights."""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import smplx
import torch
import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
import cv2
import pickle
from posedreamer.utils.paths import WEIGHTS_DIR


def render_front_back(vertices, faces, vertices_colors):
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.visual.vertex_colors = vertices_colors
    scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=np.ones(3))
    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_node)

    cam_front = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose_front = np.eye(4)
    cam_pose_front[:3, 3] = [0, 0, 2.5]  
    scene.add(cam_front, pose=cam_pose_front)

    r = pyrender.OffscreenRenderer(640, 480)

    color_front, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color_front = color_front[:, :, :3]

    scene.clear()  
    scene.add(mesh_node)  

    cam_back = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose_back = np.eye(4)
    cam_pose_back[:3, 3] = [0, 0, -2.5]  
    cam_pose_back[:3, :3] = np.diag([-1, 1, -1])  
    scene.add(cam_back, pose=cam_pose_back)

    color_back, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color_back = color_back[:, :, :3]

    concat = np.concatenate([color_front, color_back], axis=1)
    return concat


def create_new_colormap(vertices):
    min_vals = vertices.min(axis=0)
    max_vals = vertices.max(axis=0)
    normalized_vertices = (vertices - min_vals) / (max_vals - min_vals + 1e-8) 
    colormap = (normalized_vertices * 255).astype(np.uint8)  
    np.save(str(WEIGHTS_DIR / "new_colormap_smpl.npy"), colormap)  # Save the colormap for later use
    with open(str(WEIGHTS_DIR / "model_transfer/smpl2smplx_deftrafo_setup.pkl"), "rb") as f:
        smpl2smplx = pickle.load(f)
    smpl2smplx = smpl2smplx["mtx"].todense()
    smpl2smplx = np.array(smpl2smplx, dtype=np.float32)
    num_verts = smpl2smplx.shape[1] // 2
    smpl2smplx = smpl2smplx[:, :num_verts]
    smpl2smplx = torch.from_numpy(smpl2smplx)
    print(vertices.shape, smpl2smplx.shape)
    verts_smplx = torch.einsum('mn,bni->bmi', [smpl2smplx, torch.tensor(vertices)[None, :, :]])
    verts_smplx = verts_smplx.numpy()[0]
    print(verts_smplx.shape)
    min_vals = verts_smplx.min(axis=0)
    max_vals = verts_smplx.max(axis=0)
    normalized_vertices = (verts_smplx - min_vals) / (max_vals - min_vals + 1e-8) 
    colormap = (normalized_vertices * 255).astype(np.uint8)  
    np.save(str(WEIGHTS_DIR / "new_colormap_smplx.npy"), colormap)
    return colormap, verts_smplx


def get_smpl_vertex_embedding():
    embed_path = str(WEIGHTS_DIR / "mds_d=256.npy")
    embed_map, _ = np.load(embed_path, allow_pickle=True)  
    embed_map = torch.tensor(embed_map).float()[:, 0]
    embed_map -= embed_map.min()
    embed_map /= embed_map.max()
    return embed_map.cpu().numpy()


model_path = str(WEIGHTS_DIR)  
model = smplx.create(model_path, model_type='smpl', gender='neutral', use_pca=False)
smplx_model = smplx.create(
        str(WEIGHTS_DIR / "SMPLX_NEUTRAL.npz"),
        model_type="smplx",
        gender="neutral", use_face_contour=False,
        num_betas=10, flat_hand_mean=False,
        num_expression_coeffs=10,
        ext="npz", use_pca=False
)
body_pose = torch.zeros([1, 69])
angle_rad = -np.pi / 2.3 
body_pose[0, 47] = angle_rad   
body_pose[0, 50] = -angle_rad   
# Spread the legs slightly: the colormap maps each vertex's normalized
# X/Y/Z position in this pose to RGB, so with the feet apart the left and
# right foot land on different X values and get clearly different colors
# (in a closed stance they would be nearly identical).
small_angle_rad = -np.pi / 18.0
body_pose[0, 2] = - small_angle_rad
body_pose[0, 5] = small_angle_rad
pose_params = {
    'global_orient': torch.zeros([1, 3]),
    'body_pose': body_pose,
    'betas': torch.zeros([1, 10])
}
output = model(**pose_params)
vertices = output.vertices.detach().cpu().numpy().squeeze()
faces = model.faces

embedding = get_smpl_vertex_embedding()
embedding_uint8 = (embedding * 255).astype(np.uint8)  
colormap_bgr = cv2.applyColorMap(embedding_uint8, cv2.COLORMAP_JET)  
colormap_rgb = colormap_bgr[:, 0, ::-1] 
colormap_rgb = np.ascontiguousarray(colormap_rgb[:6890, ...])
vertices_colors = colormap_rgb

cse_version = render_front_back(vertices, faces, vertices_colors)

new_colors, new_vertices = create_new_colormap(vertices)

new_version = render_front_back(new_vertices, smplx_model.faces, new_colors)

pose_params = {
    'global_orient': torch.zeros([1, 3]),
    'body_pose': torch.zeros([1, 63]),
    'betas': torch.zeros([1, 10])
}
output = smplx_model(**pose_params)
new_vertices = output.vertices.detach().cpu().numpy().squeeze()
new_version_old_pose = render_front_back(new_vertices, smplx_model.faces, new_colors)

comparison = np.concatenate([cse_version, new_version, new_version_old_pose], axis=0)
plt.imshow(comparison)
plt.show()