# coding=utf-8
"""Utility script to access the ntriples DBpedia dumps from python"""
# Original author: Olivier Grisel <olivier.grisel@ensta.org>
# Part of dbpediakit package <https://github.com/ogrisel/dbpediakit>
# Modifications and additions by Aris Tritas
# License: BSD

import logging
import os
import re
from bz2 import BZ2File
from collections import defaultdict
from collections import namedtuple
from random import sample
from urllib.parse import unquote

from networkx import Graph

from ..utils.math import graph_degree_stats

URL_PATTERN = (
    "http://downloads.dbpedia.org/"
    "{version}/{lang}/{archive_name}_{lang}.nt.bz2"
)
VERSION = "3.9"
LANG = "en"

TEXT_LINE_PATTERN = re.compile(r'<([^<]+?)> <[^<]+?> "(.*)"@(\w\w) .\n')
LINK_LINE_PATTERN = re.compile(r"<([^<]+?)> <([^<]+?)> <([^<]+?)> .\n")

article = namedtuple("article", ("id", "title", "text", "lang"))
link = namedtuple("link", ("source", "target"))

logger = logging.getLogger(__name__)


def fetch_archive(archive_name, folder, lang=LANG, version=VERSION):
    """Fetch the DBpedia abstracts dump and cache it locally

    Archive name is the filename part without the language, for instance:
      - long_abstracts to be parsed with extract_text
      - skos_categories to be parsed with extract_link

    """
    folder = os.path.expanduser(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)

    url = URL_PATTERN.format(**locals())
    filename = url.rsplit("/", 1)[-1]
    filename = os.path.join(folder, filename)
    if not os.path.exists(filename):
        print("Downloading {} to {}".format(url, filename))
        # for some reason curl is much faster than urllib2 and has the
        # additional benefit of progress report and streaming the data directly
        # to the hard drive
        cmd = "curl -o %s %s" % (filename, url)
        os.system(cmd)
    return filename


def extract_link(
    archive_filename,
    max_items=None,
    predicate_filter=None,
    strip_prefix="http://dbpedia.org/resource/",
    max_id_length=300,
):
    """Extract link information on the fly

    Predicate filter can be a single string or a collection of strings
    to filter out triples that don't match.

    Return a generator of link(source, target) named tuples.

    """
    reader = BZ2File if archive_filename.endswith(".bz2") else open

    current_line_number = 0
    extracted = 0

    if predicate_filter:
        if isinstance(predicate_filter, str):
            predicate_filter = {predicate_filter}
        else:
            predicate_filter = set(predicate_filter)

    with reader(archive_filename, "rb") as f:
        for line in f:
            line = line.decode("utf-8")
            current_line_number += 1
            if max_items is not None and extracted > max_items:
                break
            if current_line_number % 500000 == 0:
                logger.info("Decoding line %d", current_line_number)
            m = LINK_LINE_PATTERN.match(line)
            if m is None:
                if TEXT_LINE_PATTERN.match(line) is None:
                    logger.warning(
                        "Invalid line %d, skipping.", current_line_number
                    )
                continue
            predicate = m.group(2)
            if predicate_filter and predicate not in predicate_filter:
                continue
            source = m.group(1)
            target = m.group(3)
            if strip_prefix is not None:
                source = source[len(strip_prefix) :]
                target = target[len(strip_prefix) :]
            if max_id_length is not None and (
                len(source) > max_id_length or len(target) > max_id_length
            ):
                logger.warning(
                    "Skipping line %d, with len(source) = %d and"
                    " len(target) = %d",
                    current_line_number,
                    len(source),
                    len(target),
                )
                continue

            yield link(source, target)
            extracted += 1


def extract_text(
    archive_filename,
    max_items=None,
    min_length=300,
    strip_prefix="http://dbpedia.org/resource/",
    max_id_length=300,
):
    """Extract and decode text literals on the fly

    Return a generator of article(id, title, text) named tuples:
    - id is the raw DBpedia id of the resource (without the resource prefix).
    - title is the decoded id that should match the Wikipedia title of the
      article.
    - text is the first paragraph of the Wikipedia article without any markup.
    - lang is the language code of the text literal

    """
    reader = BZ2File if archive_filename.endswith(".bz2") else open

    current_line_number = 0
    extracted = 0

    with reader(archive_filename, "rb") as f:
        for line in f:
            # if isinstance(line, bytes):
            line = line.decode("utf-8")
            current_line_number += 1
            if max_items is not None and extracted > max_items:
                break
            if current_line_number % 500000 == 0:
                logger.info("Decoding line %d", current_line_number)
            m = TEXT_LINE_PATTERN.match(line)
            if m is None:
                if LINK_LINE_PATTERN.match(line) is None:
                    logger.warning(
                        "Invalid line %d, skipping.", current_line_number
                    )
                continue
            id = m.group(1)
            if strip_prefix:
                id = id[len(strip_prefix) :]
            if max_id_length is not None and len(id) > max_id_length:
                logger.warning(
                    "Skipping line %d, with id with length %d",
                    current_line_number,
                    len(id),
                )
                continue
            title = unquote(id).replace("_", " ")
            text = m.group(2)  # .decode('unicode-escape')
            if len(text) < min_length:
                continue
            lang = m.group(3)

            yield article(id, title, text, lang)
            extracted += 1


# --- Generators


def long_abstracts_generator(data_dir, max_items=None):
    """

    :return:
    """
    generator = extract_text(
        fetch_archive("long_abstracts", data_dir), max_items=max_items
    )
    return generator


def article_categories_generator(data_dir, max_items=None):
    """ Load article categories tuple generator
    :param data_dir:
    :param max_items:
    :return:
    """
    generator = extract_link(
        fetch_archive("article_categories", data_dir),
        max_items=max_items,
        predicate_filter="http://purl.org/dc/terms/subject",
    )
    return generator


def skos_categories_generator(data_dir, max_items=None):
    """

    :param data_dir:
    :param max_items:
    :return:
    """
    generator = extract_link(
        fetch_archive("skos_categories", data_dir),
        max_items=max_items,
        predicate_filter="http://www.w3.org/2004/02/skos/core#broader",
    )
    return generator


def redirects_generator(data_dir, max_items=None):
    """

    :param data_dir:
    :param max_items:
    :return:
    """
    generator = extract_link(
        fetch_archive("redirects", data_dir),
        max_items=max_items,
        predicate_filter="http://dbpedia.org/ontology/wikiPageRedirects",
    )
    return generator


# redirects_dict = {r.source:r.target for r in redirects_generator }


def page_links_generator(data_dir, max_items=None):
    """

    :param data_dir:
    :param max_items:
    :return:
    """
    generator = extract_link(
        fetch_archive("page_links", data_dir),
        max_items=max_items,
        predicate_filter="http://dbpedia.org/ontology/wikiPageWikiLink",
    )
    return generator


def wikipedia_articles_from_categories(data_dir, categories, n_samples):
    """ Select all articles from selected categories & their subcategories """
    category_prefix = len("Category:")
    logger.info("Loading article categories from {}".format(data_dir))
    category_articles = defaultdict(set)
    for cat in article_categories_generator(data_dir):
        category_name = cat.target[category_prefix:]
        article_name = cat.source
        category_articles[category_name].add(article_name)
    logger.info("Article categories size: {}".format(len(category_articles)))

    logger.info("Loading SKOS categories")
    subcategories = defaultdict(set)
    for broaderRelation in skos_categories_generator(data_dir):
        parent = broaderRelation.target[category_prefix:]
        child = broaderRelation.source[category_prefix:]
        subcategories[parent].add(child)
    logger.info("SKOS categories size: {}".format(len(subcategories)))

    articles = set()
    n_tot_categories = 0
    n_articles = 0
    for category in categories:
        n_subcat = 0
        n_cat_articles = 0
        if category in category_articles:
            n_cat_articles += len(set(category_articles[category]))
            articles |= set(category_articles[category])
        if category in subcategories:
            n_subcat += len(subcategories[category])
            for subcat in subcategories[category]:
                if subcat in category_articles:
                    articles |= set(category_articles[subcat])
                    n_cat_articles += len(set(category_articles[subcat]))
        logger.info(
            "Category {} has {} sub-categories, "
            "{} total articles in subtree".format(
                category, n_subcat, n_cat_articles
            )
        )
        n_tot_categories += 1 + n_subcat
        n_articles += n_cat_articles
    logger.info(
        "Total: {} categories, {} articles".format(n_tot_categories, n_articles)
    )
    # Subsample
    sample_is = set(sample(range(len(articles)), n_samples))
    articles_dict = {
        article: i for i, article in enumerate(articles) if i in sample_is
    }
    return articles_dict


def filter_wikipedia_abstracts(data_dir, articles_index):
    """
    Only keep abstracts from articles that belong to the article index
    :param data_dir: str
    :param articles_index: the subset of article that have been selected
    :return:
    abstracts: list of documents
    dict_indices: keys corresponding to the articles in the dictionary
    """
    abstracts = []
    dict_indices = []
    logger.info("Loading article abstracts")
    long_abstracts_gen = long_abstracts_generator(data_dir)
    for i, abstract in enumerate(long_abstracts_gen):
        if abstract.id in articles_index:
            dict_indices.append(articles_index[abstract.id])
            abstracts.append(abstract.text)
    return abstracts, dict_indices


def build_wikipedia_graph(data_dir, articles_index):
    """
    Recreate the graph formed by wikipedia links between articles
    :param data_dir: str
    :param articles_index: The subset of articles that we're interested in
    :return: web_graph: networkX graph object
    """
    n_tot_articles = len(articles_index)
    web_graph = Graph()
    web_graph.add_nodes_from(range(n_tot_articles))
    page_links_gen = page_links_generator(data_dir)
    for edge in page_links_gen:
        if edge.source in articles_index and edge.target in articles_index:
            web_graph.add_edge(
                articles_index[edge.source], articles_index[edge.target]
            )
    logger.info(graph_degree_stats(web_graph))
    return web_graph
