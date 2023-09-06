import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import matplotlib.pyplot as plt

def visualizer(npy_path):
    npy = np.load(npy_path)

    # 创建一个 PyVista 网格
    grid = pv.UniformGrid()
    grid.dimensions = npy.shape
    # grid.origin = (0, 0, 0)
    # grid.spacing = (1, 1, 1)
    # 将数据赋值给网格
    grid.point_arrays['values'] = npy.flatten(order='F')  # 'F' for Fortran order
    # 创建一个 Plotter
    plotter = pv.Plotter()
    # 添加网格到 Plotter
    plotter.add_mesh(grid, cmap='viridis')
    # 显示可视化
    plotter.show()


def visualizer2(npy_path):
    npy = np.load(npy_path)

    # 创建一个 PyVista 网格
    grid = pv.UniformGrid()
    grid.dimensions = npy.shape
    # 将数据赋值给网格
    grid.point_arrays['values'] = npy.flatten(order='F')  # 'F' for Fortran order

    # 创建一个自定义colormap
    colors = [(0, 0, 1, 0), (0, 0, 1, 1)]  # R,G,B,A
    n_bins = 2  # Discretizes the interpolation into bins
    cmap_name = 'custom1'
    custom_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=5)

    # 创建一个 Plotter
    plotter = pv.Plotter()

    # 添加网格到 Plotter
    plotter.add_mesh(grid, cmap=custom_cmap)

    # 显示可视化
    plotter.show()


if __name__ == '__main__':
    # npy_path = '/home/oem/Downloads/code/output/refine_input.npy'
    # # visualizer(npy_path)
    # refine_input = np.load(npy_path)
    # refine_input_squeeze = refine_input.squeeze()
    # refine_input_0 = refine_input_squeeze[0]
    # print(f'refine input shape: {refine_input_0.shape}')

    proj_full_path = '/home/oem/Downloads/code/output/pred_proj_sph.npy'
    proj_full = np.load(proj_full_path)
    print(f'proj_full shape: {proj_full.shape}')

    visualizer2(proj_full_path)