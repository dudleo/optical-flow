import torch
import numpy as np

transforms_dtype = torch.float32
transforms_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rgb_to_grayscale(x):
    # x: Bx3xHxW

    # rgb_weights: 3x1x1
    # https://en.wikipedia.org/wiki/Luma_%28video%29

    rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140],
                               dtype=transforms_dtype,
                               device=transforms_device)

    rgb_weights = rgb_weights.view(-1, 1, 1)

    x = x * rgb_weights

    x = torch.sum(x, dim=1, keepdim=True)

    return x

# input: flow: torch.tensor 2xHxW
# output: flow_rgb: numpy.ndarray 3xHxW
def flow2rgb(flow, max_value=300):
    flow_map_np = flow.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)

def neighbors_to_channels(x, patch_size):
    # input: BxCxHxW
    # output: BxP*P*CxHxW

    kernels = torch.eye(patch_size**2,
                        dtype=transforms_dtype,
                        device=transforms_device).view(patch_size**2, 1, patch_size, patch_size)

    # kernels: P*Px1xPxP : out_channels x in_channels x H x W
    kernels = kernels.repeat(x.size(1), 1, 1, 1)
    # kernels: P*Px1xPxP : out_channels x in_channels x H x W

    x = torch.nn.functional.conv2d(input=x, weight=kernels, padding=int((patch_size-1)/2), groups=x.size(1))

    return x


def census_transform(x, patch_size):
    '''
    census transform:
    input: rgb image
    output: difference for each pixel to its neighbors 7x7
    1. rgb to gray: bxhxwxc -> bxhxwx1
    2. neighbor intensities as channels: bxhxwx1 -> bxhxwx7*7 (padding with zeros)
    3. difference calculation: L1 / sqrt(0.81 + L1^2): bxhxwx7*7 (coefficient from DDFlow)
    '''

    # x: Bx3xHxW
    x = rgb_to_grayscale(x)

    # x: Bx1xHxW
    x = neighbors_to_channels(x, patch_size=patch_size)

    # x: BxP^2xHxW - Bx1xHxW
    abs_dist_per_pixel_per_neighbor = torch.abs(x - x[:, 24].unsqueeze(1))

    # L1: BxP^2xHxW
    dist_per_pixel_per_neighbor = abs_dist_per_pixel_per_neighbor / torch.sqrt(0.81 + abs_dist_per_pixel_per_neighbor**2)
    # neighbor_dist: BxP^2xHxW
    # neighbor_dist in [0, 1]

    return dist_per_pixel_per_neighbor

def soft_hamming_distance(x1, x2):

    '''
    soft hamming distance:
    input: census transformed images bxhxwxk
    output: difference between census transforms per pixel
    1. difference calculation per pixel, per features: L2 / (0.1 + L2)
    2. summation over features: bxhxwxk -> bxhxwx1
    '''

    #x1, x2: BxCxHxW

    squared_dist_per_pixel_per_feature = (x1-x2)**2
    # squared_dist_per_pixel_per_feature: BxCxHxW

    dist_per_pixel_per_feature = squared_dist_per_pixel_per_feature / (0.1 + squared_dist_per_pixel_per_feature)
    # dist_per_pixel_per_feature: BxCxHxW

    dist_per_pixel = torch.sum(dist_per_pixel_per_feature, dim=1)
    # dist_per_pixel_per: BxHxW

    return dist_per_pixel

def census_loss(x1, x2, patch_size):
    '''
    census loss:
    1. hamming distance from census transformed rgb images
    2. robust loss for per pixel hamming distance: (|diff|+0.01)^0.4   (as in DDFlow)
    3. per pixel multiplication with zero mask at border s.t. every loss value close to border = 0
    4. sum over all pixel and divide by number of pixel which were not zeroed out: sum(per_pixel_loss)/ (num_pixels + 1e-6)
    '''

    # x1, x2: Bx3xHxW
    x1_census = census_transform(x1, patch_size)
    x2_census = census_transform(x2, patch_size)
    #x1_census, x2_census: Bxpatch_size^2xHxW

    soft_hamming_dist_per_pixel = soft_hamming_distance(x1_census, x2_census)
    # soft_hamming_dist: BxHxW

    robust_soft_hamming_dist_per_pixel = (soft_hamming_dist_per_pixel + 0.01)**(0.4)
    # robust_soft_hamming_dist_per_pixel: BxHxW

    mask = torch.zeros((robust_soft_hamming_dist_per_pixel.size()[1:]),
                       device=transforms_device,
                       dtype=transforms_dtype)

    pad = int((patch_size-1)/2)
    mask[pad:-pad, pad:-pad] = 1.0

    mask = mask.repeat(robust_soft_hamming_dist_per_pixel.size(0), 1, 1)
    # mask: BxHxW

    mask_total_weight = torch.sum(mask, dim=(0, 1, 2))

    #q: why does uflow stop gradient computation for mask in mask_total_weight, but not for mask in general?
    
    return torch.sum(robust_soft_hamming_dist_per_pixel * mask, dim=(0, 1, 2)) / (mask_total_weight + 1e-6)

def l1_loss(x1, x2):
    return torch.sum(torch.abs(x1-x2)) / x1.size(0)
