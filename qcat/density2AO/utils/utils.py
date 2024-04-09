import plotly.graph_objects as go
import numpy as np
import os
from .logging import setup_logger
from loguru import logger
from scipy.ndimage import zoom

setup_logger(level='INFO')

def visualize_3D(data: np.ndarray,
                 level = None,
                 scale = None):
    # Generate a 3D grid
    if level is None:
        level = data.max() * 0.5
    if scale is None:
        scale = (0.5, 0.5, 0.5)
    data = zoom(data, scale, mode='grid-wrap')

    x, y, z = np.mgrid[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]]

    # Define the isosurface plot with automatic level selection
    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=data.flatten(),
        isomin=level * 0.01,
        isomax=level,
        opacity=0.1,
        surface_count=17,
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    # Customize layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        margin=dict(t=0, l=0, b=0),  # Tighten the layout
    )
    return fig

def data2vasp(data:np.ndarray,
              filename = None):
    # get the current working directory
    current_working_directory = os.getcwd()
    if filename is None:
        filename = "CHGCAR"
    fname = os.path.join(current_working_directory, filename)
    nxyz = data.shape
    with open(fname, 'w') as file_obj:
        file_obj.writelines(" ".join([str(num) for num in nxyz]) + '\n')
        count = 0
        for iz in range(nxyz[2]):
            for iy in range(nxyz[1]):
                for ix in range(nxyz[0]):
                    file_obj.write(f"{data[ix, iy, iz]:^15.5e}")
                    count += 1
                    if count % 5 == 0:
                        file_obj.write("\n")

def error(data_test: np.ndarray,
          data_ref: np.ndarray,
          method: str = 'mape'):
    if not data_test.shape == data_ref.shape:
        logger.error(f"Shape mismatch: {data_test.shape} vs {data_ref.shape}")
        raise AssertionError
    methods = ['mape', 'smape']
    if not method in methods:
        logger.error(f"Method {method} not implemented")
        raise NotImplementedError

    if len(data_test.shape) == 3:
        data_test = data_test[np.newaxis, :, :, :]
        data_ref = data_ref[np.newaxis, :, :, :]

    err = None
    if method == 'mape':
        err = np.mean(np.mean(np.abs(data_test - data_ref), axis=(1, 2, 3)) / np.mean(data_ref, axis=(1, 2, 3)))
    elif method == 'smape':
        err = np.mean(np.mean(np.abs(data_test - data_ref), axis=(1, 2, 3)) / (0.5 * np.mean(data_ref, axis=(1, 2, 3)) +  0.5 * np.mean(data_test, axis=(1, 2, 3))))
    else:
        logger.error(f"Method {method} not implemented")
        raise NotImplementedError
    return err
