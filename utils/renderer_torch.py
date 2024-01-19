# from pytorch3d.renderer import (
#     OpenGLPerspectiveCameras, look_at_view_transform,
#     RasterizationSettings, BlendParams, PointLights,
#     MeshRenderer, MeshRasterizer, HardPhongShader, SoftSilhouetteShader, PerspectiveCameras, TexturesVertex, SoftPhongShader
# )
# # from pytorch3d.renderer.mesh.shader import SoftSilhouetteShader
# from pytorch3d.renderer.cameras import PerspectiveCameras #SfMPerspectiveCameras, OpenGLRealPerspectiveCameras
# #from pytorch3d.renderer.real_camera import OpenGLRealPerspectiveCameras
# import torch
# from pytorch3d.io import load_obj
# from pytorch3d.structures import Meshes
# from pytorch3d.transforms import Rotate, Translate
import matplotlib.pyplot as plt
import cv2
import numpy as np

COLORS = {
    # colorblind/print/copy safe:
    'blue': [0.85882353, 0.74117647, 0.65098039],
    'pink': [.7, .7, .9],
    'mint': [204/255., 229/255., 166/255.],
    'mint2': [ 223/255., 229/255.,202/255.],
    'green': [201/255., 216/255., 153/255.],
    'green2': [164/255., 221/255., 171/255.],
    'red': [114/255., 128/255., 251/255.],
    'orange': [97/255., 174/255., 253/255.],
    'yellow': [ 154/255., 230/255.,  250/255.]
}

# if __name__ == '__main__':
#     # img=Image.open('round.png')
#     # img=transparent_back(img)
#     # img.save('round2.png')
def human_render(inp_verts, inp_faces, inp_R=None, inp_T=None, f=None, cx=None, cy=None, background=None, viz=False, render_silhouette=False, render_HardPhong=True, mesh_color='pink'):
    # img = None
    # mask = None
    # background = np.ones((256,256))
    # f = 1000
    color = COLORS[mesh_color]
    h, w, c = background.shape
    image_size = max(w,h)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verts = inp_verts
    faces = inp_faces

    # Initialize each vertex to be white in color.
    vmin = torch.min(verts, axis=0)[0] 
    box = 2 
    tmp_verts = (verts - vmin) / box
    color = torch.Tensor(color)
    verts_rgb = color.repeat([verts.shape[0], 1])
    # verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)tmp_verts[None] #
    textures = TexturesVertex(verts_features=verts_rgb[None].to(device))

  
    inp_mesh = Meshes(
        verts=[verts.to(device)],   
        faces=[faces.to(device)], textures=textures
    ) 
    
    # Initialize an OpenGL perspective camera.
    #R, T = look_at_view_transform(3, 50, 0, device=device)
    R = inp_R.cuda()
    T = inp_T.cuda()
    focal = torch.Tensor([f])[None]
    principal_point = torch.Tensor([cx, cy])[None]
    ZNEAR = 0.01
    ZFAR = 10.0
    # cameras =  OpenGLRealPerspectiveCameras(device=device, focal_length=focal,  principal_point=principal_point, R=R, T=T, w=256, h=256, znear=ZNEAR,
    #     zfar=ZFAR)
    cameras = PerspectiveCameras(device=device, focal_length=f,  principal_point=((cx, cy),), R=R, T=T, image_size=((h, w),))
    lights = PointLights(device=device, location=[[0.0, 0.0, -0.1]])

    if render_HardPhong:
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
        )## hard
        image_ref = renderer(inp_mesh)  
        img = image_ref.squeeze()[:,:,:3].cpu().numpy() # * 255
        # bg = np.zeros_like(img)
        # bg[np.where(img != 1)] =  img[np.where(img != 1)] #* 255
        # cv2.imshow('t', img)
        # cv2.waitKey()
        # cv2.imwrite('t3.png', bg)
        # plt.figure('test')
        # plt.imshow(img)
        # plt.grid("off")
        # plt.axis("off")
        # plt.show()
       
        # temp_img = img * 255
        # background[np.where(img != 1)] =  img[np.where(img != 1)] * 255
    else:
        img = np.ones_like(background)
    if render_silhouette:
        blend_params = BlendParams(sigma=1e-5, gamma=1e-5)
        silhouette_raster_settings = RasterizationSettings(
            image_size=image_size,  # longer side or scaled longer side
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,#np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=150,  # the nearest faces_per_pixel points along the z-axis.
            bin_size=0
        )
        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=silhouette_raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )
        silhouette = silhouette_renderer(inp_mesh)
        silhouette = silhouette.cpu().numpy()
        mask = silhouette.squeeze()[...,3] #* 255 [..., 3]
        mask[np.where(mask!=0)] = 1.
    else:
        mask = np.ones_like(background)
    if viz:
        temp_img = img * 255
        background[np.where(img != 1)] =  temp_img[np.where(img != 1)] 
        if render_HardPhong:
            cv2.imshow('render', background)
        if render_silhouette:
            cv2.imshow('mask', mask)
        cv2.waitKey()
    return background, mask, img

if __name__ == '__main__':
    verts, faces_idx, _ = load_obj("./smpl.obj")#teapot
    faces = faces_idx.verts_idx
    R = torch.eye(3)[None]
    T = torch.ones(1,3)
    human_render(verts, faces, inp_R=R , inp_T=T, viz=True)
