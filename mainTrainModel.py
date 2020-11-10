import numpy as np
import matplotlib.pyplot as plt
import torch
import importlib

from CustomClasses import dataset, models, regression
import CustomClasses.utils.utils_functions as utils_functions
import CustomClasses.utils.lrp_analysis as lrp_analysis
import CustomClasses.utils.plot as c_plt

if __name__ == '__main__':
    root_dir = './Data/'
    M = 5
    g = np.arange(0., 2, 0.1)
    kappa = np.arange(-0.3, 0.27, 0.02)

    # ds = dataset.YukawaDataset
    ds = dataset.YukawaDatasetClassification

    train_ds = ds(root_dir, kappa, g, M)
    test_ds = ds(root_dir, kappa, g, M, load_test=True)

    cnn = models.CNN_Classification(train_ds.out_shape)

    loss = regression.train(cnn, train_ds, learning_rate=1e-3, break_after_fraction=0.1, epochs=0)

    cnn.eval()
    with torch.no_grad():
        path_store = 'models/test/'
        # utils_functions.store_model(cnn, path_store)
        # cnn2 = torch.load(path_store + 'parameters.pth')

        # importlib.reload(regression)
        regression.visualize_performance(cnn, test_ds)
        regression.visualize_performance(cnn, train_ds)

        dataset.plot_dataset(train_ds)

    # lrp_analysis.plot_importance(cnn2, test_ds, batch_size=24)
    # c_plt.all_filter3d(cnn2)

