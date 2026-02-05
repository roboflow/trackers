# Contributing to Trackers

Thank you for your interest in contributing to the Trackers library! Your help—whether it’s fixing bugs, improving documentation, or adding new algorithms—is essential to the success of the project. We’re building this library with the goal of making state-of-the-art object tracking accessible under a fully open license.

## Table of Contents

1. [How to Contribute](#how-to-contribute)
2. [Running Tests](#running-tests)
3. [CLA Signing](#cla-signing)
4. [Clean Room Requirements](#clean-room-requirements)
5. [Google-Style Docstrings and Type Hints](#google-style-docstrings-and-type-hints)
6. [Reporting Bugs](#reporting-bugs)
7. [License](#license)

## How to Contribute

Contributions come in many forms: improving features, fixing bugs, suggesting ideas, improving documentation, or adding new tracking methods. Here’s a high-level overview to get you started:

1. [Fork the Repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo): Click the “Fork” button on our GitHub page to create your own copy.

2. [Clone Locally](https://docs.github.com/en/enterprise-server@3.11/repositories/creating-and-managing-repositories/cloning-a-repository): Download your fork to your local development environment.

3. [Create a Branch](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop): Use a descriptive name to create a new branch:

    ```bash
    git checkout -b feature/your-descriptive-name
    ```

4. Develop Your Changes: Make your updates, ensuring your commit messages clearly describe your modifications.

5. [Commit and Push](https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop): Run:

    ```bash
    git add .
    git commit -m "A brief description of your changes"
    git push -u origin your-descriptive-name
    ```

6. [Open a Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request): Submit your pull request against the main development branch. Please detail your changes and link any related issues.

Before merging, check that all tests pass and that your changes adhere to our development and documentation standards.

## Running Tests

Install development dependencies:

```bash
uv sync --group dev
```

1. **Unit Tests**: Run the standard test suite:

    ```bash
    uv run pytest
    ```

2. **Doctests**: Run only doctests from docstrings:

    ```bash
    uv run pytest --doctest-modules trackers/ --ignore=test/
    ```

3. **Integration Tests**: Integration tests validate that our evaluation metrics match TrackEval's results on real MOT datasets. Run integration tests (requires downloading ~50MB of test data):

    ```bash
    uv run pytest -m integration
    ```

4. **All Tests**: Run all tests including integration:

    ```bash
    uv run pytest -m ""
    ```

## CLA Signing

In order to maintain the integrity of our project, every pull request must include a signed Contributor License Agreement (CLA). This confirms that your contributions are properly licensed under our Apache 2.0 License. After opening your pull request, simply add a comment stating:

```
I have read the CLA Document and I sign the CLA.
```

This step is essential before any merge can occur.

## Clean Room Requirements

Trackers package is developed under the Apache 2.0 license, which allows for wide adoption, commercial use, and integration with other open-source tools. However, many object tracking methods released alongside academic papers are published under more restrictive licenses (GPL, AGPL, etc.), which limit redistribution or usage in commercial contexts.

To ensure Trackers remains fully open and legally safe to use:

- All algorithms must be clean room re-implementations, meaning they are developed from scratch without referencing restricted source code.
- You must not copy, adapt, or even consult source code under restrictive licenses.

You can use the following as reference:

- The original academic papers that describe the algorithm.
- Existing implementations released under permissive open-source licenses (Apache 2.0, MIT, BSD, etc.).

If in doubt about whether a license is compatible, please ask before proceeding. By contributing to this project and signing the CLA, you confirm that your work complies with these guidelines and that you understand the importance of maintaining a clean licensing chain.

## Google-Style Docstrings and Type Hints

For clarity and maintainability, any new functions or classes must include [Google-style docstrings](https://google.github.io/styleguide/pyguide.html) and use Python type hints. Type hints are mandatory in all function definitions, ensuring explicit parameter and return type declarations. These docstrings should clearly explain parameters, return types, and provide usage examples when applicable.

For example:

```python
def sample_function(param1: int, param2: int = 10) -> bool:
    """
    Provides a brief description of function behavior.

    Args:
        param1 (int): Explanation of the first parameter.
        param2 (int): Explanation of the second parameter, defaulting to 10.

    Returns:
        bool: True if the operation succeeds, otherwise False.

    Examples:
        >>> sample_function(5, 10)
        True
    """
    return param1 == param2
```

Following this pattern helps ensure consistency throughout the codebase.

## Reporting Bugs

Bug reports are vital for continued improvement. When reporting an issue, please include a clear, minimal reproducible example that demonstrates the problem. Detailed bug reports assist us in swiftly diagnosing and addressing issues.

## License

By contributing to Trackers, you agree that your contributions will be licensed under the Apache 2.0 License as specified in our [LICENSE](/LICENSE) file.

Thank you for helping us build a reliable, open-source tracking library. We’re excited to collaborate with you!
