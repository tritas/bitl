# coding=utf-8
""" Export tools for visualization, reporting and re-usability.
"""
import os
from jinja2 import Template

HERE = os.path.abspath(__file__)


def _template_path(template):
    return os.path.join(HERE, '..', 'evaluation', 'templates', template)


def render_simple(results, template):
    return Template(_template_path(template)).render(results)


def render_table(results, template, is_synthetic=False):
    """
    Using Jinja2 to render html, tex and whatever other format we want.

    Parameters
    ----------
    results: TBD - metrics object or dict?
    template: filepath
    is_synthetic: bool

    Returns
    -------
    table: object

    References
    ----------
    http://jinja.pocoo.org/docs/
    """
    headers = ["Algorithm", "Runtime"]
    if is_synthetic:
        headers.append("Final Regret")
    else:
        headers.extend(["Bagged mean payoff", "Summed mean payoff"])

    template = Template(template)
    table = template.render(headers=headers, results=results)
    return table


def create_notebook(algorithm, environment):
    """ Produce Jupyter notebook with all the necessary imports
    to be able to run and experiment with an algorithm. """
    raise NotImplementedError
