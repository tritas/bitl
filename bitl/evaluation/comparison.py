# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
import sys
from time import time

from .evaluation import PolicyEvaluation


class PolicyComparison(PolicyEvaluation):
    """
    An object to run the evaluation on multiple policies
    """

    def __init__(self, policies, rewards, signal,
                 evaluation_kernel=None, n_runs=100,
                 horizon=10000, random_seed=1, n_jobs=1, verbose=0):
        self.policies = policies
        super(PolicyComparison, self).__init__(
            rewards=rewards,
            signal=signal,
            evaluation_kernel=evaluation_kernel,
            n_runs=n_runs,
            horizon=horizon,
            random_seed=random_seed,
            n_jobs=n_jobs,
            verbose=verbose)

    def replay_all(self):
        """ Runs simulations in parallel """
        t0 = time()
        for p in self.policies:
            self.replay(p)
        sys.stdout.write("Comparison took {:.3f} seconds\n"
                         .format(time() - t0))

    def output_html(self):
        """ TODO: Output metrics as HTML """
        raise NotImplementedError

    def output_tex(self):
        """ TODO: Output metrics as LaTeX """
        raise NotImplementedError

    @staticmethod
    def render_table(results):
        """ Using Jinja2 to render an HTML table """
        headers = ['Algorithm', 'Runtime']
        # if is_synthetic:
        #     headers.append('Final Regret')
        # else:
        #     headers.extend(['Bagged mean payoff', 'Summed mean payoff'])
        table = '''
<table class="table table-bordered">
    <thead>
    <tr>
    {% for header in headers %}
    <th>
        {{ header }}
    </th>
    {% endfor %}
    </tr>
    </thead>
    <tbody>
    {% for row in results %}
    <tr class="row-elt">
    {% for elt in row %}
        {% if loop.first %}
        <th>
        {{ elt }}
        </th>
        {% else %}
        <td>
        {{ elt }}
        </td>
        {% endif %}
     {% endfor %}
    </tr>
    {% endfor %}
    </tbody>
</table>
        '''
        return headers, table
