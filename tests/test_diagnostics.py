#!/usr/bin/env python3
import numpy as np


def test_tensorboard() -> None:

    import matplotlib.pyplot as plt

    from pyapes.tools.diagnostics import TensorBoardTracker

    save_dir = "./tests/tmp_dir/"

    tracker = TensorBoardTracker(save_dir, True)

    for i in range(10):

        tracker.add_scalar("test_scalar", i, i)

        x = np.arange(-3.0, 4.001, 0.1)
        y = np.arange(-4.0, 3.001, 0.1)
        X, Y = np.meshgrid(x, y)

        Z1 = np.exp(-(X**2) - Y**2)
        Z2 = np.exp(-((X - 1) ** 2) - (Y - 1) ** 2)
        noise = np.random.rand(*Z1.shape)

        Z = (0.9 * Z1 - 0.5 * Z2) * 2 + noise

        fig, ax = plt.subplots(ncols=1)
        ax.pcolormesh(Z)

        tracker.add_figure("test_fig", fig, i)
        tracker.flush()

    tracker = TensorBoardTracker(save_dir, False)

    tracker.clear_data()


def test_save_vtk():

    from tests.test_variables import test_get_box_333
    from pyapes.tools.diagnostics import SaveDictionary

    mesh = test_get_box_333(False)

    save_data = SaveDictionary(mesh, "./tests/test_data/")

    save_data.save_data(mesh.vars["U"].masks, "test_mask")


if __name__ == "__main__":
    test_tensorboard()
    test_save_vtk()
