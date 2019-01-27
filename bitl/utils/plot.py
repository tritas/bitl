# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def singleplot(data, horizon, title, filepath, xy_labels=None):
    """
    Simple plot of a data array.
    :param data: Data array
    :param horizon: Length of the x-axis
    :param title: Figure title
    :param xy_labels: Axis labels
    :param filepath: Absolute path to save figure on the filesystem
    """
    x_axis = np.arange(0.0, horizon, 1.0)
    plt.plot(x_axis, data)
    plt.title(title)
    if xy_labels:
        plt.xlabel(xy_labels[0])
        plt.ylabel(xy_labels[1])
    plt.savefig(filepath)
    plt.clf()


# List of (algo_name, data), horizon, file descriptor
def multiplot(lines_tuple, horizon, title, filepath, xy_labels=None):
    # Plotting mean reward as a function of time
    """
    Plot multiple graphs on the same figure
    :param lines_tuple:
        List of tuples each containing (name, data)
    :param horizon: Length of the x-axis
    :param title: Figure title
    :param xy_labels: Axis labels
    :param filepath: Absolute path to save figure on the filesystem
    """
    x_axis = np.arange(0.0, horizon, 1.0)
    for alg_name, data in lines_tuple:
        plt.plot(x_axis, data, label='{}'.format(str(alg_name)))
    plt.title(title)
    plt.xlabel(xy_labels[0])
    plt.ylabel(xy_labels[1])
    plt.legend()
    plt.savefig(filepath)
    plt.clf()


def plot_tsne_embeddings(matrix, title, filepath, verbose):
    """
    Computes the t-SNE 2D embedding of a precomputed similarity matrix
    and saves it to the filesystem.
    :param matrix:
        Precomputed square pairwise distance matrix
    :param title: str
        Title of the figure
    :param filepath: path, str
    :param verbose: bool
    """
    from sklearn.manifold import TSNE
    tsne = TSNE(verbose=verbose, metric='precomputed')
    embeddings = tsne.fit_transform(matrix)
    plt.scatter(embeddings[:, 0], embeddings[:, 1])
    plt.title(title)
    plt.savefig(filepath)
    plt.clf()


def save_graph_plot(graph, path, title="Graph Connectivity",
                    include_labels=True):
    """
    Draws a NetworkX graph and saves it to the filesystem
    :param graph: Graph instance
    :param path: Path to saved figure
    :param title: String to name the figure
    :param include_labels: Whether to include node and edge labels or not
    """
    nx.draw(graph, with_labels=include_labels)
    plt.title(title)
    plt.savefig(path)
    plt.clf()


def save_eigenvalues_plot(eigenvalues, absolute_path, title):
    """
    Sorts and saves the plot of an array
    :param title: Title of the figure
    :param eigenvalues: Numpy array of to plot values from
    :param absolute_path: Absolute path to figure
    """
    plt.plot(np.sort(eigenvalues))
    plt.title(title)
    plt.savefig(absolute_path)
    plt.clf()
