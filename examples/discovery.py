# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
import argparse
import logging
import re
import warnings
from os import makedirs
from os.path import exists, expanduser, join
from time import strftime, time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from bitl.policies.good_oful import GoodOFUL
from bitl.policies.loss_prop_ts import LossPropagationTS
from bitl.policies.oracle import Oracle, SpectralOracle
from bitl.policies.random import RandomWalkDiscovery
from bitl.policies.topic_discovery import TopicDiscovery
from joblib import dump, load
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from bitl.datasets.wikipedia import build_wikipedia_graph
from bitl.datasets.wikipedia import filter_wikipedia_abstracts
from bitl.datasets.wikipedia import wikipedia_articles_from_categories
from bitl.utils.math import graph_laplacian_eig, kullback_leibler_matrix

# --- Experience verbosity
warnings.simplefilter("ignore")

# --- Corpus: Wikipedia
# XXX: These should be valid Wikipedia categories (not checked)
DEFAULT_CATEGORIES = [
    'Gastronomy', 'Water_sports', 'World_War_II',
    'Television', 'Linguistics', 'Sociology',
    'Sociological_theories', 'Sociological_terminology',
    'Robotics', 'Artificial_intelligence', 'Chess',
    'Spaceflight', 'Economics', 'Restaurants'
]
DEFAULT_DATA_DIR = expanduser(join('~', 'data', 'wikipedia'))
DEFAULT_OUTPUT_DIR = join(DEFAULT_DATA_DIR, 'discovery')

# --- Bar plots
bar_width = 0.55
opacity = 0.8
# Set seed
rng_seed = 1337
np.random.seed(rng_seed)


def parse_args():
    n_cats = len(DEFAULT_CATEGORIES)
    """ Parses input arguments. """
    parser = argparse.ArgumentParser(
        description="Learning to discover interesting content on Wikipedia")
    parser.add_argument('-d', '--data_dir',
                        type=str, default=DEFAULT_DATA_DIR,
                        help='data folder (default:~/data/wikipedia/)')
    parser.add_argument(
        '-o', '--output_dir',
        type=str, default=DEFAULT_OUTPUT_DIR,
        help='output folder (default:~/data/wikipedia/discovery/)')
    parser.add_argument('-v', '--verbose',
                        type=int, default=1,
                        help='level of verbosity for training (default:1)')
    parser.add_argument('-i', '--interests',
                        type=int, default=int(n_cats / 3),
                        help='Number of interesting topics (default:n/3)')
    parser.add_argument('-u', '--dislikes',
                        type=int, default=int(n_cats / 3),
                        help='Number of uninteresting topics (default:n/3)')
    parser.add_argument('-b', '--beta',
                        type=float, default=1,
                        help='distance function transform param (default:1)')
    parser.add_argument('-s', '--sparsify',
                        action='store_false',
                        help='avoid having a complete graph (default:True)')
    parser.add_argument('-m', '--max_per_cat',
                        type=str, default=500,
                        help='maximum articles per category (default:500)')
    parser.add_argument('-n', '--samples',
                        type=int, default=10000,
                        help='Number of samples to keep (default:10000)')
    parser.add_argument('-f', '--features',
                        type=int, default=None,
                        help='Number of features to keep (default:all)')
    parser.add_argument(
        '-t', '--topics',
        type=str, default=DEFAULT_CATEGORIES, nargs='+',
        help='Topics to sample from (default:{})'.format(DEFAULT_CATEGORIES))
    parser.add_argument('-k', '--n_topics',
                        type=int, default=n_cats,
                        help='Number of topics (default:{})'.format(n_cats))
    parser.add_argument('-p', '--procs',
                        type=int, default=-1,
                        help='how many procs to use (default:all)')
    parser.add_argument('-r', '--runs',
                        type=int, default=1,
                        help='how many runs to make (default:1)')
    parser.add_argument('--noise',
                        type=float, default=0.2,
                        help='reward additive noise variance (default:0.2)')
    parser.add_argument('--tfidf',
                        action='store_true',
                        help='use TF-IDF representation (default:False)')
    parser.add_argument(
        '--tsne',
        action='store_true',
        help='plot t-SNE embeddings (default:False)')
    parser.add_argument(
        '--extra-plots', action='store_true',
        help='extra plots that take long to compute (default:False)')
    parsedArgs = parser.parse_args()
    return parsedArgs


# --- Utility functions

def build_path(data_dir, fn, ext='.bz2'):
    """ Build absolute path from the data directory.
    :param fn: filename for the saved file
    :param ext: extension (default:bz2)
    :return: an absolute path
    """
    if not fn.endswith(ext):
        fn += ext
    fp = join(data_dir, fn)
    return fp


# --- Figures output
def figure_path(fig_dir, title, ext='png'):
    """ Build path to figure given title using the figure directory
    :param title: filename for the saved figure
    :param ext: extension (default:bz2)
    :return: an absolute path
    """
    fig_fn = '.'.join([title, ext])
    fp = join(fig_dir, fig_fn)
    return fp


def print_top_words(model, feature_names, n_top_words=50):
    """

    :param model:
    :param feature_names:
    :param n_top_words:  Words to show from each infered topic
    """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


# --- Plot functions
def plot_interesting(fig_dir, items_found):
    """
    Plot the evolution of interesting items found over time
    :param fig_dir:  str, path
    :param items_found: An array representing a time series of items found
    """
    plt.title('Interesting items found (N={})'.format(items_found))
    fig = plt.gcf()
    fig.set_size_inches((8, 8), forward=True)
    plt.xlabel('Timesteps')
    plt.ylabel('No. Items')
    plt.legend(loc='lower right', fontsize='large', fancybox=True)
    fn = 'Interesting items found ({}, {}) {}'.format(
        items_found, n_topics, strftime('%d-%m %H-%M-%S'))
    plt.savefig(figure_path(fig_dir, fn))
    plt.clf()


def learn_dictionary(topic_distribution, fig_dir, n_features, n_procs):
    """ TODO: Compare:
     np.eye() initialization & normal KMeans (for small corpora)
     Multiple runs (they're cheap) -> Keep best fit """

    n_samples, n_topics = topic_distribution.shape
    # Bar plot params
    n_topics_arange = np.arange(n_topics)

    label = 'LDA Topics_{}_{}_{}'.format(n_features, n_samples, n_topics)
    topic_weights = np.sort(topic_distribution.sum(axis=0) /
                            topic_distribution.sum())[::-1]
    plt.bar(n_topics_arange,
            topic_weights,
            bar_width,
            alpha=opacity,
            color='b',
            label=label)

    print('\nClustering the topic distribution..')
    t0 = time()
    if n_samples > 2e5:
        kmeans = MiniBatchKMeans(n_clusters=n_topics,
                                 n_init=100,
                                 max_no_improvement=30,
                                 verbose=0,
                                 random_state=rng_seed)
    else:
        kmeans = KMeans(n_clusters=n_topics,
                        n_init=100,
                        n_jobs=n_procs,
                        random_state=rng_seed)

    assignments = kmeans.fit_predict(topic_distribution)
    print('done in {:.3f}s. Final inertia = {}'.format(
        time() - t0, kmeans.inertia_))

    plt.bar(n_topics_arange + bar_width,
            np.sort(kmeans.cluster_centers_.sum(axis=0) /
                    kmeans.cluster_centers_.sum())[::-1],
            bar_width,
            alpha=opacity,
            color='r',
            label='Kmeans Clusters')
    plt.xlabel('Clusters')
    plt.ylabel('Relative weight')
    plt.title('Original Topics and Clusters weights ({} topics)'
              .format(n_topics))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path(fig_dir, 'LDA Topic Distribution'))
    plt.clf()

    return kmeans.cluster_centers_, assignments


def make_theoretic_means(n_topics,
                         n_interests,
                         n_dislikes,
                         topic_assignments,
                         fig_dir,
                         noise_variance=0.5):
    """ Compute the theoretic mean rewards:
    choose an item as interesting with some low probability iff its topic
    belongs to the interesting topics set and ever lower if it doesn't"""
    corpus_size = len(topic_assignments)
    rewards = np.zeros(corpus_size, dtype=np.float64)
    draws = np.random.rand(corpus_size)
    interest_indicator = np.zeros(corpus_size, dtype=np.uint8)

    interests = np.random.choice(
        n_topics - 1, size=n_interests, replace=False)
    dislikes_arr = np.random.choice(
        n_topics - n_interests - 1, size=n_dislikes, replace=False)

    topic_set = set(range(n_topics))
    rest = topic_set - set(interests)
    dislikes = np.array(list(rest))[dislikes_arr]
    # Thresholds (can be regarded as Bernouillis)
    high_interest_pr = 0.5
    low_interest_pr = 0.1
    lowest_interest_pr = 0.05
    # Top-n item topics
    print('Interesting topics: {}; Uninteresting topics: {}'.format(
        interests, dislikes))
    print('\nComputing theoretic means...')
    t0 = time()
    for i, item_assignment in enumerate(topic_assignments):
        # Likeable items - reward=1 w.h.p
        if item_assignment in interests:
            if draws[i] < high_interest_pr:
                interest_indicator[i] = 1
                rewards[i] = 1
        # Dislikeable items - reward=-1 w.h.p
        elif item_assignment in dislikes:
            if draws[i] < lowest_interest_pr:
                interest_indicator[i] = 1
                rewards[i] = 1
            else:
                rewards[i] = -1
        # Any item - reward=1 w. low pr.
        else:
            if draws[i] < low_interest_pr:
                interest_indicator[i] = 1
                rewards[i] = 1

    rewards += np.random.randn(corpus_size) * noise_variance
    rewards = np.clip(rewards, -1, 1)

    print('done in {:.3f}s.'.format(time() - t0))
    print('\nNumber of deterministically interesting items: {}/{}'
          .format(interest_indicator.sum(), corpus_size))
    '''
    means_stats = pd.Series(artificial_means)
    print(means_stats.describe(percentiles=np.arange(0, 1, 0.05)))

    plt.hist(artificial_means, bins=20)
    plt.savefig(figure_path('Means distribution histogram'))
    plt.clf()
    '''
    plt.bar(np.arange(corpus_size),
            np.sort(rewards),
            alpha=opacity,
            color='g')
    plt.xlabel('Reward mean')
    plt.ylabel('No. Items')
    plt.title('Items mean ({}/{} interesting/total items, {} topics)'
              .format(interest_indicator.sum(), corpus_size, n_topics))
    plt.tight_layout()
    plt.savefig(figure_path(fig_dir, 'Mean rewards bar plot'))
    plt.clf()

    return rewards, interest_indicator


def infer_topic_distrib(X, samples_permutation, n_features, verbose, n_procs):
    """ Define a CountVectorizer and latent Dirichlet allocation model,
    perform training and inference on the samples dataset.
    :param X: list of documents' bag-of-words
    :param samples_permutation: array
    """
    token_pattern = re.compile(r'(?u)\b[a-zA-Z]{2}\w+\b')

    def tokenizr(doc):
        """ Tokenize all words in a document, ignoring stopwords
        :param doc:
        :return:
        """
        stem_func = SnowballStemmer('english', ignore_stopwords=True).stem
        return list(map(stem_func, token_pattern.findall(doc)))

    tf_vectorizer = CountVectorizer(max_df=0.9, max_features=n_features,
                                    tokenizer=tokenizr,
                                    token_pattern=r"(?u)\b[a-zA-Z]{2}\w+\b",
                                    stop_words='english')

    # Parameters tuned for the Wikipedia corpus
    # according to [Hoffman et al. 2010]
    lda = LatentDirichletAllocation(
        n_topics=n_topics,
        evaluate_every=3,
        learning_decay=0.5,
        batch_size=1024,
        learning_offset=1024.,
        random_state=1,
        verbose=verbose,
        n_jobs=n_procs)

    # Vectorize documents to extract word counts
    print('\nExtracting term frequency features for LDA...')
    t0 = time()
    tf = tf_vectorizer.fit_transform(X)
    print('done in {:.3f}s. Number of samples={}, features={}'
          .format(time() - t0, tf.shape[0], tf.shape[1]))

    # Fit LDA and normalize returned gamma distribution
    # Extract topic mixture vector
    t0 = time()
    doc_topic_distrib = lda.fit_transform(tf)
    doc_topic_distrib /= doc_topic_distrib.sum(axis=1)[:, np.newaxis]
    print('done in {:.3f}s.'.format(time() - t0))

    sorted_indx = np.argsort(samples_permutation)
    doc_topic_distrib = doc_topic_distrib[sorted_indx, :]

    print('\nTopics in LDA model:')
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names)

    return doc_topic_distrib


def maybe_mkdir(d):
    if not exists(d):
        makedirs(d)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    output_dir = args.output_dir
    data_dir = args.data_dir
    log_dir = join(output_dir, 'logs')
    models_dir = join(output_dir, 'models')
    figures_dir = join(output_dir, 'figs')
    logfile = join(log_dir, 'wikipedia_discovery.log')
    # Prepare filesystem & logging
    maybe_mkdir(log_dir)
    maybe_mkdir(data_dir)
    maybe_mkdir(figures_dir)
    logging.basicConfig(filename=logfile, level=logging.INFO)
    # --- Experience hyper-parameters
    verbose = args.verbose
    # Plots that take long to compute
    extra_plots = args.extra_plots
    n_runs = args.runs
    n_procs = args.procs
    # No. of most frequent features to use for vocabulary extraction
    n_features = args.features
    # Which categories (and their 1-depth subcategories) to include
    categories = args.topics
    max_articles_per_category = args.max_per_cat
    n_topics = len(categories)
    n_articles = args.samples
    # --- Virtual user's model
    n_interests = args.interests
    n_dislikes = args.dislikes
    reward_noise_var = args.noise
    # --- Which algos to test
    loss_prop_discovery = False
    # Compute and use spectral properties
    # (eigenvectors are not yet GoodOFUL features)
    spectral = False
    # Compute kernel matrix and inverse
    # (default is False because of heavy compute time)
    kernel = False

    # --- Semantic hyperparameters
    '''
    If sparsify is set to True, values such that:
    KLinv = exp(- beta * KL) < relevance_threshold
    are set to zero to induce sparsity in the similarity graph
    and avoid having a complete graph.

    The relevance threshold can be set depending on the value of beta
    and therefore the resulting histogram of the KLinv matrix distribution
    '''
    beta = args.beta
    sparsify = args.sparsify
    relevance_threshold = 0.1
    interest_threshold = 0.3
    # Import all data from Wikipedia dumps
    max_imports = None

    # --- Feature extraction
    articles_path = build_path(
        data_dir, 'wikipedia_articles_{}_{}'.format(n_articles, n_topics))
    try:
        articles_dict = load(articles_path)
    except FileNotFoundError:
        articles_dict = wikipedia_articles_from_categories(
            data_dir, categories, n_articles)
        dump(articles_dict, articles_path)

    web_graph_path = build_path(
        data_dir, 'wikipedia_graph_{}'.format(n_articles))
    try:
        web_graph = load(web_graph_path)
    except FileNotFoundError:
        web_graph = build_wikipedia_graph(data_dir, articles_dict)
        dump(web_graph, web_graph_path)

    features_path = build_path(
        data_dir, 'wikipedia_features_{}'.format(n_articles))
    try:
        doc_topic_dist = load(features_path)
        _, n_topics = doc_topic_dist.shape
    except FileNotFoundError:
        abstracts, permutation = filter_wikipedia_abstracts(
            data_dir, articles_dict)
        doc_topic_dist = infer_topic_distrib(
            abstracts, permutation, n_features, verbose, n_procs)
        dump(doc_topic_dist, features_path)

    centroids, cluster_assignments = learn_dictionary(
        doc_topic_dist, figures_dir, n_features, n_procs)

    eigen_path = build_path(data_dir, 'wiki_eigen_{}'.format(n_articles))
    try:
        V, W = load(eigen_path)
    except FileNotFoundError:
        V, W = graph_laplacian_eig(web_graph)
        dump((V, W), eigen_path)

    # KL divergence between nodes (Kernel matrix) passed as side-information
    KL, KLinv = kullback_leibler_matrix(
        doc_topic_dist, beta, relevance_threshold)

    synthetic_means, is_interesting_array = make_theoretic_means(
        n_topics, n_interests, n_dislikes, cluster_assignments, figures_dir,
        noise_variance=reward_noise_var)
    '''
    print('\nComputing confidence parameter..')
    t0 = time()
    C = np.linalg.norm(np.dot(artificial_means.T,
    np.dot(V, artificial_means)))
    print('done in {:.3f}s. C = {}\n'.format(time() - t0, C))
    '''
    # --------- Algorithms parametrization

    policy_lst = [
        Oracle(synthetic_means),
        RandomWalkDiscovery(n_items=n_articles)
    ]

    lambdas = [0.01, 0.033, 0.01, 0.33, 1, 3.3, 10]  # regularization
    delta = 0.01  # confidence
    R = 0.01
    theta_star = np.random.randn(n_topics)  # + W.shape[1]

    discovery_params = {
        'debug': verbose,
        'lambda': 0.01,
        'delta': delta,
        'gamma': 0.2,
        'depth': 6,
        'graph': web_graph,
        'sim_mat': KLinv,
        'means': synthetic_means,
        'seed': np.random.randint(n_articles),
        'R': R,
        'C': 10,
        'P': 50,
        'k': n_topics,
        'items': doc_topic_dist[:n_articles],
        'eig': (V, W.copy())
    }

    if spectral:
        manifold_spectral = discovery_params.copy()
        manifold_spectral['eig'] = (V, W)
        manifold_spectral['alg'] = 'Manifold'
        policy_lst.append(SpectralOracle(**manifold_spectral))

    if kernel:
        semantic_spectral = discovery_params.copy()
        topic_graph = nx.Graph(KLinv)
        Y, Z = graph_laplacian_eig(topic_graph)
        semantic_spectral['eig'] = (Y, Z)
        semantic_spectral['alg'] = 'Semantic'
        policy_lst.append(SpectralOracle(**semantic_spectral))

    if spectral and kernel:
        ts_unseeded = {
            'graph': web_graph,
            'seed': [],
            'assignments': cluster_assignments[:n_articles],
            'sim_mat': KLinv,
            'discounted': False
        }

        ''' Pick closest (scalar product argmax) feature vect to each cluster protoype:
        seed with the feature vect + cluster reward
        seeds = []
        for i in topic_set:
            pass

        ts_seeded = ts_unseeded.copy()
        ts_seeded['seed'] = seeds
        '''
        ts_unseeded_disc = ts_unseeded.copy()
        ts_unseeded_disc['discounted'] = True
        policy_lst.extend([
            LossPropagationTS(**ts_unseeded),
            LossPropagationTS(**ts_unseeded_disc)
        ])
        # SemanticLossPropagationTS(**ts_seeded)

    if loss_prop_discovery:
        policy_lst.append(TopicDiscovery(**discovery_params))

    good_oful_params = {
        'items': doc_topic_dist[:n_articles],
        'means': centroids,
        'assignments': cluster_assignments[:n_articles],
        'c': 2.0,
        'S': 0.5,
        'C': 1.0,
        'R': 0.5,
        'alpha': 0.2,
        'delta': delta,
        'lambda': 1.0,
        'interest_threshold': interest_threshold
    }
    policy_lst.append(GoodOFUL(**good_oful_params))

'''
for a in np.arange(0.10, 1.0, 0.40).round(1):
    oful_params = good_oful_params.copy()
    oful_params['alpha'] = a
    algs_run_lst.append(GoodOFUL(**oful_params))
'''
