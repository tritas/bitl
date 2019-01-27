# -*- coding: utf-8 -*-

=====
Todos
=====

Synthetic datasets
------------------
* Reward blocks: row/col multiplicity
* Matrix sensing techiques (like universal singular value sampling)
* Item features: sparsify, clusters of interesting and uninteresting actions
* Realistic user clusters & features.
Each user cluster distribution could be considered e.g. as a normal distribution centered at x
+ some skewed distribution centered at x + eps
 (http://stackoverflow.com/questions/5884768/skew-normal-distribution-in-scipy),
  (https://en.wikipedia.org/wiki/Skew_normal_distribution)

Real-world datasets
-------------------

* Enron corpus
* Yandex
* All recommendations (and their context)

Evaluation
----------
* Discovery kernel
* Offline evaluation using counterfactual information
* Develop problem suites

Reward distributions
--------------------

Documentation, testing and examples
-----------------------------------

* Annotate algorithms with author information and proper defaults
* Test environments
* Sanity check on known regret lower/upper bounds

Utilities
---------

* Run policy code from C/C++, Matlab, Java, etc.
* Export to iPython notebook
* Logging
* Plot enveloppe from multiple runs std.dev.

Performance
-----------

* Cache simulation data (datasets, features) and results to disk
* Cythonize
