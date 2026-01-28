Contributing to LoS Estimator
=============================

We welcome contributions to the LoS Estimator project! This guide will help you get started with development and submitting contributions.

Getting Started
---------------

**Setting Up Your Development Environment**

1. Clone the repository:

   .. code-block:: bash

       git clone git@git.rwth-aachen.de:jrc-combine/los-estimator.git
       cd los-estimator

2. Create a virtual environment:

   .. code-block:: bash

       python -m venv .venv
       
       # On Windows
       .\.venv\Scripts\activate
       
       # On Linux/macOS
       source .venv/bin/activate

3. Install the package in editable mode with development dependencies:

   .. code-block:: bash

       pip install -e ".[dev,docs]"

Development Workflow
--------------------

**Code Style**

We use Black for code formatting and follow PEP 8. Before submitting a PR:

.. code-block:: bash

    black los_estimator/

**Type Hints**

We encourage type hints for better code quality. Check your code:

.. code-block:: bash

    mypy los_estimator/

**Running Tests**

Run the test suite to ensure your changes work:

.. code-block:: bash

    pytest tests/

For coverage report:

.. code-block:: bash

    pytest --cov=los_estimator tests/

**Building Documentation**

Documentation is built with Sphinx using reStructuredText:

.. code-block:: bash

    cd docs
    
    # On Windows
    .\make.bat html
    
    # On Linux/macOS
    make html

View the built documentation in ``docs/build/html/index.html``.

**Important Files and Directories**

- ``los_estimator/`` - Main package source code
- ``tests/`` - Unit and integration tests
- ``docs/source/`` - Documentation source (RST format)
- ``examples/`` - Example scripts and data
- ``pyproject.toml`` - Package configuration and dependencies
- ``setup.py`` - Minimal build configuration

Making Changes
--------------

**Before You Start**

1. Check existing issues and PRs to avoid duplicating work
2. Create a descriptive issue if one doesn't exist
3. For major changes, discuss in an issue before implementing

**Creating a Feature Branch**

.. code-block:: bash

    git checkout -b feature/your-feature-name
    # or for bug fixes
    git checkout -b fix/your-bug-fix-name

**Commit Guidelines**

- Use clear, descriptive commit messages
- Keep commits atomic (one logical change per commit)
- Reference related issues in commits: ``Fixes #123`` or ``Related to #456``

Example:

.. code-block:: bash

    git commit -m "Add new distribution model for exponential fitting

    - Implement exponential distribution class
    - Add unit tests for new model
    - Update documentation
    
    Fixes #123"

**Documentation Comments**

Code should include docstrings following Google style:

.. code-block:: python

    def estimate_los(admissions, occupancy):
        """Estimate length of stay distribution from admission and occupancy data.

        This function uses deconvolution methods to estimate the probability
        distribution of patient length of stay.

        Args:
            admissions (np.ndarray): Daily admission counts
            occupancy (np.ndarray): Daily occupancy counts

        Returns:
            dict: Estimated distribution parameters

        Raises:
            ValueError: If input arrays have mismatched lengths
            TypeError: If inputs are not numpy arrays
        """

Submitting a Pull Request
-------------------------

1. Push your branch to the repository:

   .. code-block:: bash

       git push origin feature/your-feature-name

2. Open a Pull Request (PR) on GitLab with:

   - Clear title describing the change
   - Description explaining what and why
   - Reference to related issues
   - Screenshots or examples if applicable

3. Address review feedback:

   - Respond to comments professionally
   - Make requested changes in new commits
   - Push updates (no need to force push on open PR)

**PR Checklist**

Before submitting, verify:

- [ ] Type hints are added where appropriate
- [ ] Tests are added for new functionality
- [ ] Tests pass locally (``pytest``)
- [ ] Documentation is updated if needed
- [ ] CHANGELOG.md is updated with your changes
- [ ] No unnecessary files are committed

Code Review Process
-------------------

All PRs require review before merging. Reviewers will check:

- Code quality and adherence to style
- Test coverage and pass rates
- Documentation clarity
- Compatibility with existing code
- Performance implications

Be patient and constructive during review. Questions are meant to improve the code, not criticize.

Reporting Issues
----------------

**Bug Reports**

Include:

- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Python version, OS, and package version
- Relevant error messages and traceback
- (Optional) Minimal reproducible example

**Feature Requests**

Include:

- Use case and motivation
- Proposed solution (if you have one)
- Alternative approaches considered
- (Optional) Example code showing the desired API

Support and Questions
---------------------

For questions or discussions:

- Use GitLab Issues for technical discussions
- Check existing documentation and issues first
- Be specific and provide context

Code of Conduct
---------------

We are committed to providing a welcoming and inclusive environment. 
All contributors are expected to be respectful and constructive in their interactions.

.. note::
    Please report any Code of Conduct violations to the project maintainers.

Licensing
---------

By contributing to LoS Estimator, you agree that your contributions will be 
licensed under the GPLv3 License. See LICENSE.md for details.

Recognition
-----------

Contributors will be recognized in:

- Project README
- Release notes
- GitHub/GitLab contributor list

Thank you for contributing! 🎉
