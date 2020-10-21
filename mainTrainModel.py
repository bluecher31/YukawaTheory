import numpy as np
import matplotlib.pyplot as plt
import torch
import importlib

from CustomClasses import dataset, models, regression
import CustomClasses.utils.utils_functions as utils_functions
import CustomClasses.utils.lrp_analysis as lrp_analysis

if __name__ == '__main__':
    root_dir = './Data/'
    M = 5.
    g = np.arange(0.1, 2, 0.1)
    kappa = np.arange(-0.2, 0.19, 0.02)
    batch_size = 32

    # ds = dataset.YukawaDataset
    ds = dataset.YukawaDatasetClassification

    train_ds = ds(root_dir, kappa, g, M)
    test_ds = ds(root_dir, kappa, g, M, load_test=True)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    cnn = models.CNN_Classification(train_ds.out_shape)

    loss = []
    for i in range(5):
        print(f'epoch = {i}')
        temp = regression.train(cnn, train_loader, learning_rate=1e-3, break_after_fraction=1.)
        loss.append(temp)


    cnn.eval()
    with torch.no_grad():
        path_store = 'models/test/'
        # utils_functions.store_model(cnn, path_store)
        # cnn2 = torch.load(path_store + 'parameters.pth')

        # importlib.reload(regression)
        regression.visualize_performance(cnn, test_ds)
        regression.visualize_performance(cnn, train_ds)


        dataset.plot_dataset(test_ds)
        # plt.show()

    # lrp_analysis.plot_importance(cnn, test_ds)
