import jax
import jax.numpy as jnp
from tqdm import tqdm

num_vertices = 10000
num_poses = 2000
vertices = jnp.zeros((num_vertices, 3))
poses = jnp.zeros((num_poses, 3, 3))
fx,fy,cx,cy = 100.0, 100.0, 100.0, 100.0
image = jnp.ones((640, 480, 4))

@jax.jit
def per_vertex(vertex, pose, image):
    # 12386 fps
    # transformed_vertex = pose[0] + vertex
    
    # 95fps
    transformed_vertex = pose @ vertex
    
    x = fx * transformed_vertex[0] / (transformed_vertex[2]) + cx
    y = fy * transformed_vertex[1] / (transformed_vertex[2]) + cy
    projected_pixel = jnp.array([y, x]).astype(jnp.int32)
    rgbd_pixel = image[projected_pixel[0], projected_pixel[1]]
    return rgbd_pixel.sum()

@jax.jit
def per_pose(pose, image):
    scores = jax.vmap(per_vertex, in_axes=(0, None, None))(vertices, pose, image)
    return scores.sum()

@jax.jit
def full(image):
    pose_scores = jax.vmap(per_pose, in_axes=(0, None))(poses, image)
    return pose_scores[jnp.argmax(pose_scores)]

full(image)

for _ in tqdm(range(10000)):
    x = full(image)