import jax
import jax.numpy as jnp
import flax
from internal import math

@flax.struct.dataclass
class Surf_Ray:
    gaussians: float
    directions: float
    near: float
    far: float
    origins: float
    radii: float
    imageplane: float = None
    exposure_values: float = None
    surface_tdist: float = None

# Ray positions are (num rays, 3) (i.e. the most dense point on a ray, corresponding like to a surface)
# Scales are (num rays, 1) (i.e. the GSD from the camera of the most dense point on the ray)
# Weights are (num rays, 1) (i.e. the weight of the point on the surface)
def surface_reg_batch(PRNG, sphere, gaussians, rays, surface_tdist, surface_normals, surface_dir_reg_front = True, sample_points=64):

    ray_origins = rays.origins
    ray_dirs = rays.directions
    num_rays = ray_origins.shape[0]


    if sample_points != sphere.shape[0]:
      def sample_n_perms(key):
          return jax.random.choice(key, sphere, (sample_points,), replace=False)
      

      key_array = jax.random.split(PRNG, num_rays)

      sphere_samples = jax.vmap(sample_n_perms, in_axes=(0))(key_array)[:,None,None,...] # This should give me (num_rays, sample_points, 3)
    else:
      sphere_samples = math.random_rot_matrices(PRNG, jnp.broadcast_to(sphere, (num_rays, *sphere.shape)))[:,None,None,...]

    dirs = math.l2_normalize(sphere_samples)

    means, covs = gaussians


    basis_matrix = math.align_vectors(jnp.broadcast_to(ray_dirs[:,None,...],dirs.shape), dirs)

    tmu = (surface_tdist[...,1:]+surface_tdist[...,:-1]) / 2
    td = (surface_tdist[...,1:]-surface_tdist[...,:-1]) / 2
    sigma_r2 = rays.radii**2 * (tmu**2 / 4 + 5* td**2 / 12 - 4 * td**4 / (15 * (3 * tmu **2 + td**2)))
    

    
    # Only works with the diagonalised version (3x3):
    # max_diag_n = jnp.max(jnp.einsum('...ii->...i',covs),axis=-1, keepdims = True)
    max_diag = jnp.sqrt(2 * jnp.log(2)) * jnp.sqrt(sigma_r2)[...,None] # This is an upper bound

    sample_means_dir = jnp.broadcast_to(means, sphere_samples.shape)
    sample_means_spa = sample_means_dir + max_diag * sphere_samples

    sample_means = jnp.concatenate([sample_means_spa, sample_means_dir], axis = -2) # Stack along axis 2 (...,num_samples, 3)

    ray_origins_shift = (ray_origins[...,None,:] - sample_means_dir)
    ray_origins_rotate = jnp.einsum('...ij,...j', basis_matrix, ray_origins_shift) + sample_means_dir

    sample_origins = jnp.concatenate([ray_origins_rotate, ray_origins_rotate], axis = -2)


    # Only works with the diagonalised version (3x3):
    covs_rotate = jnp.einsum('...ij,...jk', jnp.einsum('...ji,...jk', basis_matrix, covs), basis_matrix)    
    sample_covs = jnp.concatenate([covs_rotate, covs_rotate], axis = -3) # Concatenate long axis 3 (..., num_samples, 3, 3)

    if surface_dir_reg_front:
      dir_normals_dot = jnp.sum(dirs * surface_normals, axis = -1, keepdims = True)
      dirsff = jnp.where(dir_normals_dot > 0, -dirs, dirs)
      sample_dirs = jnp.concatenate([dirs, dirsff], axis = -2)

    else:

      sample_dirs = jnp.concatenate([dirs, dirs], axis = -2)

    sample_gaussians = (sample_means, sample_covs)

    sample_batch = Surf_Ray(gaussians=sample_gaussians, directions = sample_dirs, near=rays.near, far = rays.far, imageplane=rays.imageplane, exposure_values = rays.exposure_values, surface_tdist = surface_tdist, origins = sample_origins, radii = rays.radii)

    return sample_batch


def surface_reg_losses(ray_results, batch_positions, batch_directions, config):
  ## Spatial Reg
  spa_sample_density = ray_results['density'][...,:config.surface_samples]
  spa_adj_normals = ray_results['normals'][...,:config.surface_samples,:]

  # jax.debug.print("ROUGHNESS {x}", x=jnp.mean(1/ray_results['roughness'][...,config.surface_samples:,:]))

  # We calc this here as opposed to trying to propagate gradients back via the original sample.
  normals = jnp.mean(ray_results['normals'][...,config.surface_samples:,:],axis=-2,keepdims=True) # Stop Grad Else OOM
  spa_sample_weight = 1 - math.safe_exp(-spa_sample_density)[...,None]

  sample_points = batch_positions[...,:config.surface_samples,:] - batch_positions[...,config.surface_samples:,:]

  normals_consist = jnp.sum(spa_adj_normals * normals, axis = -1, keepdims = True)


  spoint_normal_dot = 1 - jnp.sum(math.l2_normalize(sample_points) * normals, axis = -1, keepdims= True)
  dir_normal_dot = jnp.sum(batch_directions[...,config.surface_samples:,:] * normals, axis = -1, keepdims = True)


  spatial_reg = spa_sample_weight * (jnp.abs(spoint_normal_dot)) #+ (1 - spa_sample_weight) * (1- jnp.abs(spoint_normal_dot)) # Move stuff closer to the surface or reorientate surface.

  # normal_reg = math.l2_length(spa_adj_normals - normals)**2 
  normal_reg = (1 - jnp.abs(normals_consist)) / 2 # If they're pointing in the same direction, normals_consist = 1 then all g.

  dir_spherical_coords = math.cartesian_to_spherical(sample_points)

  ## Directional Reg
  # If no diffuse, we are dealing with view-dependent only term - we can still enforce total var
  # diffuse = jnp.mean(ray_results['diffuse'][...,config.surface_samples:,:], axis=-2, keepdims=True)
  dir_specular = ray_results['specular'][...,config.surface_samples:,:]
  dir_mean_specular = jnp.mean(dir_specular, axis = -2, keepdims=True)
  var_specular = jnp.var(dir_specular, axis = -2, keepdims = True)
  # var_specular = jnp.mean(math.l2_length(dir_specular - dir_mean_specular), axis = -2, keepdims=True)
  
  # jax.debug.print("VAR_SPECULAR {x}", x = jnp.mean(var_specular))
  # jax.debug.print("VAR_SPECULAR2 {x}", x = jnp.mean(jnp.var(dir_specular, axis = -2)))

  # r_val = math.l2_length(jnp.sum(math.l2_normalize(dir_specular), axis = -2, keepdims = True))/(config.surface_samples)
  # r_val = math.l2_length(jnp.sum(dir_specular, axis = -2, keepdims = True))/(config.surface_samples)

  # kappa_est = r_val * jnp.maximum((3 - r_val**2),jnp.finfo(jnp.float32).eps)  / jnp.maximum(1 - r_val**2, jnp.finfo(jnp.float32).eps)
  # exp_kappa = math.safe_exp(kappa_est)
  # jax.debug.print("KAPPA_EST {x}", x = jnp.mean(kappa_est))
  dir_max_specular = jnp.max(dir_specular,axis=-2, keepdims=True)

  # jax.debug.print("MEAN/MAX {x}", x = jnp.mean(dir_mean_specular/dir_max_specular))



  if ray_results['diffuse'] is None:
    # Total var over LF component
    # bias_reg = 0.0
    bias_reg = jnp.mean(jnp.where(dir_normal_dot > 0, (2-dir_normal_dot)*math.l2_length(dir_specular)**2, math.l2_length(dir_specular)**2), axis=-2, keepdims=True)
  else:
     
     
  # We calc this here as opposed to trying to propagate gradients back via the original sample.
    # diffuse = jnp.mean(ray_results['diffuse'][...,config.surface_samples:,:],axis=-2,keepdims=True)
    # dir_specular = ray_results['specular'][...,config.surface_samples:,:]
    # dir_min_specular = jax.lax.stop_gradient(jnp.min(dir_specular,axis=-2, keepdims=True)) # Stop Grad on Min
    # # Cosine distance between specular term minimum and diffuse
    # bias_reg = math.l2_length(dir_min_specular)*(1 + math.cosine_similarity(diffuse, dir_min_specular))/2

    # bias_reg = (3 - math.l2_length(diffuse))
    # bias_reg = jnp.sum(jnp.where(dir_normal_dot > 0, dir_normal_dot, 1)*dir_specular, axis=-2, keepdims=True)/jnp.sum(jnp.where(dir_normal_dot > 0, dir_normal_dot, 1))**2
    # bias_reg = jnp.mean(jnp.where(spoint_normal_dot > 0, math.cosine_similarity(diffuse, dir_specular), math.l2_length(dir_specular)**2 / 3), axis=-1)
    # bias_reg = (1 - math.l2_length(diffuse)/jnp.sqrt(3))
    # bias_reg = jnp.mean((jnp.sum(jnp.where(dir_normal_dot>0, jnp.abs(dir_normal_dot)*dir_specular, dir_specular), axis = -2, keepdims = True) / jnp.sum(jnp.where(dir_normal_dot>0, jnp.abs(dir_normal_dot), 1), axis = -2, keepdims = True)), axis = -1, keepdims=True)
    bias_reg = jnp.sum((dir_specular / jax.lax.stop_gradient(jnp.maximum(dir_max_specular, 1e-8)))**2, axis = -2, keepdims = True)
    # bias_reg = jnp.sum(jnp.min(dir_specular, axis = -2, keepdims = True)**2, axis = -2, keepdims = True)
    # bias_reg = jnp.min(dir_specular,axis=-2, keepdims=True)
    # jax.debug.print("BIAS_REG {x}", x = jnp.mean(bias_reg))
    # jax.debug.print("WTF {x}", x = jnp.mean(jnp.sum(jnp.where(dir_normal_dot>0, 1, 0), axis = -2, keepdims = True)))





  def total_var_knn(sample, sample_val, all_samples, sample_vals, k = 5):
    dists = math.cosine_similarity(sample[...,None,:], all_samples)
    neighbour_dists, idx = jax.lax.approx_max_k(dists, k = k, reduction_dimension=-2)

    neighbour_vals = jnp.take_along_axis(sample_vals, idx, axis=-2) # Only constrain current sample
    sample_tv = jnp.mean((neighbour_dists+1)/2 * jnp.mean(jnp.abs(neighbour_vals - sample_val[...,None,:]), axis = -1, keepdims=True), axis = -2, keepdims=True)
    return sample_tv
  
  total_var_fn = jax.vmap(total_var_knn,in_axes=(-2,-2,None,None),out_axes=-2)
  


  specular_reg = total_var_fn(dir_spherical_coords, dir_specular, jax.lax.stop_gradient(dir_spherical_coords), jax.lax.stop_gradient(dir_specular))

  # jax.debug.print("SPATIAL {x}, NORMALS {n}, BIAS {b}, specular {s}, diffuse_colour {dc}, spec_mean {spec_mean}", x = jnp.mean(spatial_reg), n = jnp.mean(normal_reg), b = jnp.mean(bias_reg), s = jnp.mean(specular_reg), dc = jnp.mean(diffuse), spec_mean = jnp.mean(dir_mean_specular))



  return dict(spatial_reg = spatial_reg, 
              normal_reg = normal_reg, 
              bias_reg = bias_reg,
              specular_reg = specular_reg,)

# Hold this in memory initially, take random subset of hemisphere, rotate + scale, then run through network.
# TODO(jack) - get rid of split between halves
def uniform_sphere(PRNG, rad = 1, n = 2048, volume_sample = False):

  keys = jax.random.split(PRNG, 3)

  u = jax.random.uniform(keys[0],(n,1),minval=0,maxval=1)
  phi = jax.lax.acos(1 - 2 * u)
  theta = 2* jnp.pi * jax.random.uniform(keys[1],(n,1),minval=0,maxval=1)
  
  if volume_sample:
    randrad = jax.random.uniform(keys[2],(n,1),minval=0,maxval=rad)
  else:
    randrad = rad

  x = randrad * math.safe_sin(phi) * math.safe_cos(theta)
  y = randrad * math.safe_sin(phi) * math.safe_sin(theta)
  z = randrad * math.safe_cos(phi)


  sphere = jnp.concatenate((x,y,z), axis = -1)

  return sphere # Get rid of spare axis

def fibonacci_sphere(n = 2048, volume_sample = True):
  # Inspired by: https://ieeexplore.ieee.org/document/10361396
  # With the "uniform sample sphere" trick as employed above.

  invphi = (jnp.sqrt(5) - 1) / 2

  l = jnp.arange(1, n+1)[...,None]

  if volume_sample:
    # Space in log_2(n) shells to sample the volume
    radii = 1/jnp.log2(n) * (1 + jnp.mod(l, jnp.log2(n)))
  else:
    radii = 1

  # Eq: 8
  theta = 2*jnp.pi * jnp.mod(l * invphi, 1)
  phi = jax.lax.acos(1 - 2 * jnp.mod((2*l - 1)/ (2*n), 1))

  x = radii * math.safe_sin(phi) * math.safe_cos(theta)
  y = radii * math.safe_sin(phi) * math.safe_sin(theta)
  z = radii * math.safe_cos(phi)


  sphere = jnp.concatenate((x,y,z), axis = -1)

  return sphere # Get rid of spare axis
   
def vMF_sphere_samples(kappa=0, L = 2048, volume_sample = True):
  # From: https://ieeexplore.ieee.org/document/10361396
  i = jnp.arange(1, L+1)[...,None]
  invphi = (jnp.sqrt(5) - 1) / 2

  if volume_sample:
    # Space in log_2(n) shells to sample the volume
    radii = 1/jnp.log2(L) * (1 + jnp.mod(i, jnp.log2(L)))
  else:
    radii = 1

  if kappa == 0:
    # Use the analytical limit of the function to avoid numerical nasties
    w = ((1-2*i) + L)/L
  else:
    # Eq: 40
    w = 1 + 1/kappa * jnp.log(1 + (2*i-1)/(2*L)*(jnp.exp(-2*kappa)-1))

  # Eq: 39
  x = radii * w
  y = radii * jnp.sqrt(1-w*w)*jnp.cos(2*jnp.pi*i*invphi) 
  z = radii * jnp.sqrt(1-w*w)*jnp.sin(2*jnp.pi*i*invphi)

  sphere = jnp.concatenate((x,y,z), axis = -1)
  return sphere