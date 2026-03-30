import jax
import jax.numpy as jnp
import numpy as np

def canonical_simplex(d):
    return jnp.where(jnp.arange(d + 1)[:, None] <= d - jnp.arange(d + 1), jnp.arange(d + 1), jnp.arange(d + 1) - (d + 1))

d3_simplex = canonical_simplex(3)

def permutohedral_coordinates_single(coords):
    def alpha(i): 
        return jnp.sqrt(i / (i+1))
    d = coords.shape[-1]

    permuto_coords = [-alpha(d) * coords[...,d-1]]
    for i in range(d-1, 0, -1):
        permuto_coords.append(-alpha(i) * coords[...,i-1] + 1/alpha(i+1) * coords[..., i] + permuto_coords[-1])
    permuto_coords.append(1/alpha(1) * coords[..., 0] + permuto_coords[-1])

    return jnp.stack(permuto_coords,axis=-1)[...,::-1]

def permuto_coord_matrix(d):
  def alpha(i): 
      return jnp.sqrt(i / (i+1))
  
  matrix = jnp.zeros((d+1,d))

  # scale_factor = jnp.sqrt(1.0/6.0)*(d+1)

  for j in range(d):
      for i in range(0, j+2):
        if i == j+1:
            matrix = matrix.at[i,j].set(-alpha(j+1))
        else:
            matrix = matrix.at[i,j].set(1/alpha(j+1)-alpha(j+1))
  return matrix #scale_factor * matrix

d3_coord_matrix = permuto_coord_matrix(3)

def permutohedral_coordinates(coords):
  return jnp.einsum("ij,...j->...i", d3_coord_matrix, coords)
             
             
def barycentric_matrix(d = 3):
    matrix = jnp.zeros((d+1, d+1))
    matrix = matrix.at[-d:, -d:].set(-jnp.eye(d)[:,::-1])
    matrix = matrix.at[0,0].set(-1)
    matrix += jnp.eye(d+1)[:,::-1]
    return matrix/(d+1)

bary_matrix = barycentric_matrix()

def barycentric_coords_2(y, pos_dim = 3):
    # Create a matrix for the differences
    # Compute the barycentric coordinates using matrix multiplication
    b = jnp.einsum("ij,...j->...i", bary_matrix, y)
    b = b.at[...,0].add(1)

    # Compute the final barycentric coordinate as 1 - sum of the rest
    
    return b


def barycentric_coords(y, d = 3):
    # d = y.shape[-1] - 1

    b = []
    for i in range(d, 0, -1):
        b.append((y[...,d-i] - y[...,d-i+1])/(d+1))
    
    b.append(1 - sum(b))

    return b[::-1]

def find_permutohedral_vertices(permuto_coords, d = 3):


    down_factor = 1 / (d + 1)
    up_factor = d + 1

    #TODO(jack) Shifting by 1/2 is likely not needed?
    v = down_factor * permuto_coords

    vup = (jnp.ceil(v) * up_factor).astype(jnp.int32)
    vdown = (jnp.floor(v) * up_factor).astype(jnp.int32)
    # rd = jnp.floor(0.5 + down_factor * permuto_coords)
    # rd = jnp.floor(0.5 + permuto_coords)

    rem0 = jnp.where(vup - permuto_coords < permuto_coords - vdown, vup, vdown).astype(jnp.int32)

    rem0_sum = jnp.sum(rem0, axis=-1, keepdims=True) * down_factor
    
    


    canon_coords = permuto_coords - rem0

    # Sort in DESCENDING order
    coords_sort = canon_coords.argsort()[...,::-1].argsort() + rem0_sum.astype(jnp.int32)

    # Because we are using argsort, these will always be bounded. No need for the wrapping.
    #TODO(jack) - rewrite the below better, these are both CoOKeD! Perhaps it may be faster to just create a new var - memory is cheap?
    adjust_val = jnp.where(coords_sort < 0, d + 1, jnp.where(coords_sort > d,  - (d+1), 0))
    rem0 += adjust_val
    coords_sort += adjust_val

    simplex_broad = jnp.broadcast_to(d3_simplex, (*coords_sort.shape,d+1))

    vert = []

    for i in range(d+1):
        vert.append((rem0+jnp.take_along_axis(simplex_broad[...,i], coords_sort,axis=-1))[...,:3])

    # vertices = jnp.stack(vert,axis=-1)

    v = canon_coords * down_factor

    barycentric_weights = barycentric_coords(v)

    return barycentric_weights, vert



def gather_volume_permuto(data, locations, coordinate_order='xyz'):
  """Gather from data at locations.

  Args:
    data: A [D, H, W, C] tensor.
    locations: A [D, ..., 3] int32 tensor containing the locations to sample at.
    coordinate_order: Whether the sample locations are x,y,z or z,y,x.

  Returns:
    A [D, ..., C] tensor containing the gathered locations.
  """
  if coordinate_order == 'xyz':
    x_coordinate = locations[Ellipsis, 0]
    y_coordinate = locations[Ellipsis, 1]
    z_coordinate = locations[Ellipsis, 2]
    # w_coordinate = locations[Ellipsis, 3]
  elif coordinate_order == 'zyx':
    # w_coordinate = locations[Ellipsis, 0]
    z_coordinate = locations[Ellipsis, 0]
    y_coordinate = locations[Ellipsis, 1]
    x_coordinate = locations[Ellipsis, 2]

  # Use Advanced indexing to gather data data.
  return data[z_coordinate, y_coordinate, x_coordinate]

def permuto_lattice_sample_opt(position, pos_dim = 3):

  # matrix = permuto_coord_matrix(pos_dim)
  # elevated = jnp.einsum("ij,...j->...i", matrix, position)
  elevated = permutohedral_coordinates(position)

  # print(elevated)

  v = elevated * (1 / (pos_dim + 1))
  up = jnp.ceil(v) * (pos_dim + 1)
  down = jnp.floor(v) * (pos_dim + 1)

  rem0 = jnp.where(up - elevated < elevated - down, up, down)
  sum_val = jnp.sum(rem0, axis=-1, keepdims=True) / (pos_dim + 1)



  rank = jnp.argsort(elevated - rem0)[...,::-1].argsort()


  rank += sum_val
  adjust_val = jnp.where(rank < 0, pos_dim + 1, jnp.where(rank > pos_dim, - pos_dim - 1, 0))
  rank += adjust_val
  rem0 += adjust_val

  rank = rank.astype(jnp.int32)

  barycentric = jnp.zeros((*position.shape[:-1], pos_dim+2))

  delta = (elevated - rem0)[..., None] * (1 / (pos_dim + 1))
  indexing_arrays = [jnp.arange(dim_size).reshape((dim_size,) + (1,) * (len(barycentric.shape) - i))
                  for i, dim_size in enumerate(barycentric.shape[:-1])]


  rank_indices_pos = (pos_dim - rank)[..., None]
  rank_indices_neg = (pos_dim + 1 - rank)[..., None]


  barycentric = barycentric.at[(*indexing_arrays, rank_indices_pos)].add(delta)


  barycentric = barycentric.at[(*indexing_arrays, rank_indices_neg)].add(-delta)

  # Final addition
  barycentric = barycentric.at[..., 0].add(1 + barycentric[..., pos_dim + 1])

  i_vals = jnp.arange(pos_dim + 1, dtype=jnp.int32)


  # Create key template by broadcasting
  key_template = rem0[..., None, :].astype(jnp.int32) + i_vals[:, None]
  rank_adjustment = jnp.where(rank[..., None, :] > pos_dim - i_vals[:, None], - pos_dim - 1, 0)
  key_template += rank_adjustment
  
  return barycentric, key_template

def permuto_lattice_sample(position, pos_dim = 3):

  elevated = jnp.zeros((*position.shape[:-1], pos_dim+1))
  sm = 0
  for i in range(pos_dim, 0, -1):
      cf = position[..., i-1] * 1 / jnp.sqrt((i) * (i+1))
      elevated = elevated.at[..., i].set(sm - i * cf)
      sm += cf
  elevated = elevated.at[..., 0].set(sm)

  # print(elevated)

  sum_val = 0

  v = elevated * (1 / (pos_dim + 1))
  up = jnp.ceil(v) * (pos_dim + 1)
  down = jnp.floor(v) * (pos_dim + 1)

  rem0 = jnp.where(up - elevated < elevated - down, up, down)
  sum_val = jnp.sum(rem0, axis=-1, keepdims=True) / (pos_dim + 1)
  # print(sum_val)

  rank = jnp.zeros_like(elevated)

  for i in range(pos_dim):
      di = elevated[..., i] - rem0[..., i]
      for j in range(i+1, pos_dim+1):
          rank = rank.at[..., i].set(rank[..., i] + jnp.where(di < elevated[..., j] - rem0[..., j], 1, 0))
          rank = rank.at[..., j].set(rank[..., j] + jnp.where(di >= elevated[..., j] - rem0[..., j], 1, 0))

  # print(rank)

  rank += sum_val
  # print("------ RANK ---------")
  # print(rank)

  # rank_bcast = jnp.broadcast_to(rank, rem0.shape)
  adjust_val = jnp.where(rank < 0, pos_dim + 1, jnp.where(rank > pos_dim, - pos_dim - 1, 0))
  rank += adjust_val
  # print(rank)
  # print("------ REM0---------")
  # print(rem0)
  rem0 += adjust_val
  # print(rem0)
  # print(jnp.where(rank_bcast < 0, pos_dim + 1, jnp.where(rank_bcast > pos_dim, - pos_dim - 1, 0)))
  rank = rank.astype(jnp.int32)

  barycentric = jnp.zeros((*position.shape[:-1], pos_dim+2))

  indexing_arrays = [jnp.arange(dim_size).reshape((dim_size,) + (1,) * (len(barycentric.shape) - i - 1))
                    for i, dim_size in enumerate(barycentric.shape[:-1])]

  for i in range(pos_dim + 1):
      delta = (elevated[..., i] - rem0[..., i])[...,None] * (1 / (pos_dim + 1))

      barycentric = barycentric.at[(*indexing_arrays, (pos_dim - rank[..., i])[...,None])].add(delta)
      barycentric = barycentric.at[(*indexing_arrays, (pos_dim + 1 - rank[..., i])[...,None])].add(-delta)

  barycentric = barycentric.at[..., 0].add(1 + barycentric[..., pos_dim+1])

  keys = []
  key = jnp.zeros_like(position)

  for i in range(pos_dim + 1):
      for j in range(pos_dim):
          key = key.at[..., j].set(rem0[..., j] + i)
          key = key.at[..., j].add(jnp.where(rank[..., j] > pos_dim - i, - pos_dim - 1, 0))

      keys.append(key)
  
  return barycentric, keys




def resample_permuto(
    data,
    locations,
    grid_size,
    edge_behavior='CONSTANT_OUTSIDE',
    constant_values=0.0,
    coordinate_order='xyz',
    half_pixel_center=False,
):
  """Resamples input data at the provided locations from a volume.

  Args:
    data: A [D, H, W, C] tensor from which to sample.
    locations: A [D, ..., 3] containing floating point locations to sample data
      at. Assumes voxels centers at integer coordinates.
    edge_behavior: The behaviour for sample points outside of params.
      -CONSTANT_OUTSIDE: First pads params by 1 with constant_values in the
      x-y-z dimensions, then clamps samples to this padded tensor. The effect is
      that sample points interpolate towards the constant value just outside the
      tensor. -CLAMP: clamps to volume.
    constant_values: The constant value to use with edge_behvaior
      'CONSTANT_OUTSIDE.'
    coordinate_order: Whether the sample locations are x,y,z or z,y,x.
    method: The interpolation kernel to use, must be 'TRILINEAR' or 'NEAREST'.
    half_pixel_center: A bool that determines if half-pixel centering is used.

  Returns:
    A tensor of shape [D, ..., C] containing the sampled values.
  """

  assert len(data.shape) >= 3
  assert edge_behavior in ['CONSTANT_OUTSIDE', 'CLAMP']
  if edge_behavior == 'CONSTANT_OUTSIDE':
    data = jnp.pad(
        data,
        np.array([[1, 1], [1, 1], [1, 1]] + (data.ndim - 3) * [[0, 0]]),
        constant_values=constant_values,
    )
    locations = locations + 1.0

  # Trilinearly interpolates by finding the weighted sum of the eight corner
  # points.
  if half_pixel_center:
      locations = locations - 0.5
  
#   permuto_coords = permutohedral_coordinates(locations)

# #   simplex = canonical_simplex(locations.shape[-1])

#   weights, positions = find_permutohedral_vertices(permuto_coords)

  weights, positions = permuto_lattice_sample_opt(locations)

#   positions = []
#   weights = []
#   for i in range(4):
#      positions.append(vertices[...,i,:])
#      weights.append(bary_weights[...,i])

  max_indices = jnp.array(data.shape[:3], dtype=jnp.int32) - 1
  if coordinate_order == 'xyz':
    max_indices = jnp.flip(max_indices)

  output = jnp.zeros((*locations.shape[:-1], data.shape[-1]), dtype=data.dtype)

  # for position, weight in zip(positions, weights):
  for i in range(4):

    weight = weights[...,i]
    position = positions[...,i,:3] + 4 * grid_size

    indexes = position.astype(jnp.int32)

    indexes = jnp.maximum(indexes, 0)
    indexes = jnp.minimum(indexes, max_indices)
    gathered = gather_volume_permuto(data, indexes, coordinate_order)
    weighted_gathered = (
        gathered if weight is None else gathered * weight[Ellipsis, None]
    )

    output += weighted_gathered

  return output.astype(data.dtype)


def hash_resample_permuto(
    data, locations, half_pixel_center=True
):
  """Resamples input data at the provided locations from a hash table.

  Args:
    data: A [D, C] tensor from which to sample.
    locations: A [D, ..., 3] containing floating point locations to sample data
      at. Assumes voxels centers at integer coordinates.
    method: The interpolation kernel to use, must be 'TRILINEAR' or 'NEAREST'.
    half_pixel_center: A bool that determines if half-pixel centering is used.

  Returns:
    A tensor of shape [D, ..., C] containing the sampled values.
  """

  assert len(data.shape) == 2

  if half_pixel_center:
      locations = locations - 0.5
  
  # permuto_coords = permutohedral_coordinates(locations)

#   simplex = canonical_simplex(locations.shape[-1])

  # weights, positions = find_permutohedral_vertices(permuto_coords)
  weights, positions = permuto_lattice_sample_opt(locations)


  output = None
  # for position, weight in zip(positions, weights):
  for i in range(4):

    weight = weights[...,i]
    position = positions[...,i,:3]

    position = position.astype(jnp.int32).astype(jnp.uint32)

    # These are from Teschner 2003, and in fact pi_2 is NOT a prime! Seemingly continually overlooked in the literature.
    pi_2 = 19349663
    pi_3 = 83492791
    # pi_4 = 73856093

    # Apparently, these are better? https://planetmath.org/goodhashtableprimes
    # All primes (minimises clustering), all as far from nearest powers of 2.
    # pi_2 = 12582917
    # pi_3 = 25165843
    # pi_4 = 50331653

    data_indexes = jnp.mod(
        jnp.bitwise_xor(
            position[Ellipsis, 0],
            jnp.bitwise_xor(position[Ellipsis, 1] * pi_2, position[Ellipsis, 2] * pi_3),
        ),
        data.shape[0],
    ).astype(jnp.int32)
    gathered = data[(data_indexes,)]
    weighted_gathered = (
        gathered if weight is None else gathered * weight[Ellipsis, None]
    )
    if output is None:
      output = weighted_gathered
    else:
      output += weighted_gathered

  return output