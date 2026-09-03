Contributing to LoS Estimator
=============================

Getting Started
---------------

1. Clone the repository and create a virtual environment:

   .. code-block:: bash

       git clone git@github.com:JRC-COMBINE/los-estimator.git
       cd los-estimator
       python -m venv .venv
       .\.venv\Scripts\activate   # Windows; `source .venv/bin/activate` on Linux/macOS

2. Install in editable mode with dev/docs extras:

   .. code-block:: bash

       pip install -e ".[dev,docs]"

Development Workflow
--------------------

.. code-block:: bash

    black los_estimator/          # formatting
    mypy los_estimator/           # type checking
    pytest tests/                 # tests
    pytest --cov=los_estimator tests/   # with coverage

Build docs with Sphinx:

.. code-block:: bash

    cd docs
    .\make.bat html   # Windows; `make html` on Linux/macOS

Output goes to ``docs/build/html/index.html``.

**Layout**

- ``los_estimator/`` - package source
- ``tests/`` - integration tests
- ``docs/source/`` - Sphinx sources (RST)
- ``examples/`` - example scripts and data
- ``pyproject.toml`` / ``setup.py`` - packaging

Adding a New Distribution Kernel
---------------------------------

Distribution kernels are registered in one place,
``los_estimator/fitting/distributions.py``, and are then automatically
available to the rolling-window fit, evaluation, and visualization stages —
no other module needs to know about a new distribution. The steps below add a
log-logistic kernel (``scipy.stats.fisk``) as a worked example.

1. Add a name constant to ``DistributionTypes``:

   .. code-block:: python

       class DistributionTypes:
           ...
           LOGLOGISTIC = "loglogistic"

2. Add a ``Distribution`` entry to the ``_distributions`` dict, giving the
   PDF, the optimizer's initial values, and its bounds:

   .. code-block:: python

       from scipy.stats import fisk

       DistributionTypes.LOGLOGISTIC: Distribution(
           name=DistributionTypes.LOGLOGISTIC,
           init_values=[2, 10],
           boundaries=[(1.0, 10.0), (0.5, 60.0)],
           pdf=lambda x, c, scale: fisk.pdf(x, c=c, scale=scale),
           to_string=lambda c, scale: f"c={c}, scale={scale}",
       ),

   ``init_values`` seeds the first rolling window's optimizer; later windows
   warm-start from the previous window's fit when
   ``model_config.reuse_last_parametrization`` is enabled. ``boundaries``
   constrains ``scipy.optimize`` and should keep the initial values away from
   a bound the optimizer cannot leave (see the ``weibull``/``cauchy``/``t``
   entries in the same dict for examples of that failure mode). If the PDF
   has no usable scale parameter of its own (e.g. ``beta``, ``sentinel``),
   set ``uses_scaling=True`` so the day axis is stretched by a fitted factor
   instead.

3. Add a matching entry to ``SAMPLING_BOUNDS`` (same module), used only for
   rejection-sampling the uncertainty bands — keep it at least as wide as
   ``boundaries``:

   .. code-block:: python

       SAMPLING_BOUNDS["loglogistic"] = [(1.0, 10.0), (0.5, 60.0)]

4. Add the new name to ``model_config.distributions`` in the TOML config
   used for the run (e.g. ``default_config.toml`` or an override file):

   .. code-block:: toml

       [model_config]
       distributions = ["gaussian", "exponential", "loglogistic"]

That's it — ``fit_convolution``, ``Evaluator``, and the plotting/animation
code all look up kernels by name through ``Distributions[distro]`` /
``Distributions.generate_kernel(...)``, so a registered distribution needs no
further changes. The exception is ``compartmental``, which is fit via a
dedicated ``fit_compartmental`` path instead of the generic convolution fit
(see ``MultiSeriesFitter.fit_distro``) — only relevant if a new kernel needs
distribution-specific fitting logic rather than a plain PDF.

Adding a New Distribution Kernel
---------------------------------

Distribution kernels are registered in a single place,
``los_estimator/fitting/distributions.py``, and are then automatically
available to the rolling-window fit, evaluation, and visualization stages —
no other module needs to know about a new distribution. The steps below add a
log-logistic kernel (``scipy.stats.fisk``) as a worked example.

1. Add a name constant to ``DistributionTypes``:

   .. code-block:: python

       class DistributionTypes:
           ...
           LOGLOGISTIC = "loglogistic"

2. Add a ``Distribution`` entry to the ``_distributions`` dict, giving the
   PDF, the optimizer's initial values, and its bounds:

   .. code-block:: python

       from scipy.stats import fisk

       DistributionTypes.LOGLOGISTIC: Distribution(
           name=DistributionTypes.LOGLOGISTIC,
           init_values=[2, 10],
           boundaries=[(1.0, 10.0), (0.5, 60.0)],
           pdf=lambda x, c, scale: fisk.pdf(x, c=c, scale=scale),
           to_string=lambda c, scale: f"c={c}, scale={scale}",
       ),

   ``init_values`` seeds the first rolling window's optimizer; later windows
   warm-start from the previous window's fit when
   ``model_config.reuse_last_parametrization`` is enabled. ``boundaries``
   constrains ``scipy.optimize`` and should keep the initial values away from
   a bound the optimizer cannot leave (see the ``weibull``/``cauchy``/``t``
   entries in the same dict for examples of that failure mode). If the PDF
   has no usable scale parameter of its own (e.g. ``beta``, ``sentinel``),
   set ``uses_scaling=True`` so the day axis is stretched by a fitted factor
   instead.

3. Add a matching entry to ``SAMPLING_BOUNDS`` (same module), used only for
   rejection-sampling the uncertainty bands — keep it at least as wide as
   ``boundaries``:

   .. code-block:: python

       SAMPLING_BOUNDS["loglogistic"] = [(1.0, 10.0), (0.5, 60.0)]

4. Add the new name to ``model_config.distributions`` in the TOML config
   used for the run (e.g. ``default_config.toml`` or an override file):

   .. code-block:: toml

       [model_config]
       distributions = ["gaussian", "exponential", "loglogistic"]

That's it — ``fit_convolution``, ``Evaluator``, and the plotting/animation
code all look up kernels by name through ``Distributions[distro]`` /
``Distributions.generate_kernel(...)``, so a registered distribution needs no
further changes. The exception is ``compartmental``, which is fit via a
dedicated ``fit_compartmental`` path instead of the generic convolution fit
(see ``MultiSeriesFitter.fit_distro``) — only relevant if a new kernel needs
distribution-specific fitting logic rather than a plain PDF.

Submitting a Pull Request
-------------------------

- Keep commits atomic with clear messages; reference related issues.
- Run ``black``, ``mypy``, and ``pytest`` before opening the PR.
- Describe what changed and why; note any docs that need updating.

Licensing
---------

By contributing, you agree that your contributions will be licensed under
the GPLv3 License (see ``LICENSE.md``).
