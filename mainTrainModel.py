import numpy as np
import logging
import click
import torch
import importlib

from CustomClasses import dataset, models, regression
import CustomClasses.utils.utils_functions as utils_functions
import CustomClasses.utils.lrp_analysis as lrp_analysis
import CustomClasses.utils.plot as c_plt


@click.group()
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@main.group()
def model():
    pass


@model.command("train")
@click.option("--mass", type=int, default=5)
@click.option("--g-min", type=float, default=0.0)
@click.option("--g-max", type=float, default=2.0)
@click.option("--g-step", type=float, default=0.1)
@click.option("--k-min", type=float, default=-0.3)
@click.option("--k-max", type=float, default=0.27)
@click.option("--k-step", type=float, default=0.02)
@click.option("--data-classif/--data", default=True)
@click.option("--n-epochs", type=int, default=5)
@click.option("--lr", type=float, default=1e-3)
@click.option("--lrp/no-lrp", default=False)
@click.option("--save/no-save", default=True)
@click.pass_context
def train(mass, g_min, g_max, g_step, k_min, k_max, k_step, data_classif, n_epochs, lr, lrp, save):

    root_dir = './Data/'
    g = np.arange(g_min, g_max, g_step)
    kappa = np.arange(k_min, k_max, k_step)

    if data_classif:
        ds = dataset.YukawaDatasetClassification
    else:
        ds = dataset.YukawaDataset

    train_ds = ds(root_dir, kappa, g, mass)
    test_ds = ds(root_dir, kappa, g, mass, load_test=True)

    cnn = models.CNN_Classification(train_ds.out_shape)

    loss = regression.train(cnn, train_ds, learning_rate=lr, break_after_fraction=0.1, epochs=n_epochs)

    cnn.eval()
    with torch.no_grad():
        path_store = 'models/test/'
        if save:
            utils_functions.store_model(cnn, path_store)

        # importlib.reload(regression)
        regression.visualize_performance(cnn, test_ds)
        regression.visualize_performance(cnn, train_ds)

        dataset.plot_dataset(train_ds)

    if lrp:
        cnn2 = torch.load(path_store + 'parameters.pth')
        lrp_analysis.plot_importance(cnn2, test_ds, batch_size=24)
        c_plt.all_filter3d(cnn2)


if __name__ == "__main__":
    main()