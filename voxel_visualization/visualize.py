from voxel_visualization.util import *
from voxel_visualization.util_vtk import visualization
import numpy as np

def np_read_tensor(filename):
    """ return a 4D matrix, with dimensions point, x, y, z """

    voxels = np.load(filename)

    dims = voxels.shape
    if len(dims) == 5:
        assert dims[1] == 1
        dims = (dims[0],) + tuple(dims[2:])
    elif len(dims) == 3:
        dims = (1,) + dims
    else:
        assert len(dims) == 4
    result = np.reshape(voxels, dims)
    return result

def load_tensor(filename, varname='instance'):
    """ return a 4D matrix, with dimensions point, x, y, z """
    assert(filename[-4:] == '.mat')
    mats = loadmat(filename)
    if varname not in mats:
        print(".mat file only has these matrices:")
        for var in mats:
            print(var)
        assert(False)

    voxels = mats[varname]

    dims = voxels.shape
    #print('dims is : ',dims)
    if len(dims) == 5:
        voxels = np.squeeze(voxels, axis=4)
        dims = dims[:4]
        #assert dims[1] == 1
        #dims = (dims[0],) + tuple(dims[2:])
    elif len(dims) == 3:
        dims = (1,) + dims
    else:
        assert len(dims) == 4
    result = np.reshape(voxels, dims)
    return result

def visualizer(path):
    ind = 0
    threshold = 0.01
    connect = 3
    downsample_factor = 1
    downsample_method = 'max'
    uniform_size = 0.9
    use_colormap = "store_true"

    model_numpy = np.load(path)

    # 创建一个和proj_full形状相同的全零数组
    model_01 = np.zeros_like(model_numpy)

    # 将proj_full中非零元素的位置设置为1
    model_01[model_numpy != 0] = 1

    voxels = model_01

    if connect > 0:
        voxels_keep = (voxels >= threshold)
        voxels_keep = max_connected(voxels_keep, connect)
        voxels[np.logical_not(voxels_keep)] = 0

    if downsample_factor > 1:
        print("==> Performing downsample: factor: " + str(downsample_factor) + " method: " + downsample_method)
        voxels = downsample(voxels, downsample_factor, method=downsample_method)
        print("Done")

    visualization(voxels, threshold, title=str(ind + 1), uniform_size=uniform_size,
                  use_colormap=use_colormap)

if __name__ == '__main__':
    visualizer('/home/oem/Downloads/code/output/output_20230901_092224/batch0000/proj_depth.npy')