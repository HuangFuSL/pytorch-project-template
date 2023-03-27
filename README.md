# Python Project Template

A project template for Python projects. Inspired by [xinntao/ProjectTemplate-Python](https://github.com/xinntao/ProjectTemplate-Python).

## Features

-   [Poetry](https://github.com/python-poetry/poetry) to manage dependencies.
-   [autopep8](https://github.com/hhatto/autopep8), [isort](https://github.com/PyCQA/isort), [pylint](https://github.com/hhatto/autopep8) and [pre-commit](https://github.com/pre-commit/pre-commit) for automatic code style check.
-   [pytest](https://github.com/pytest-dev/pytest) for unit tests.
-   [poethepoet](https://github.com/nat-n/poethepoet) for custom command-line tasks.
-   Licensed under MIT license.

## Usage

**Notice**: `.vscode` directory can be removed if Visual Studio Code is not used in development.

### Setting up the project

1. Create a repository using this template.
2. Edit project settings in `pyproject.toml`.
3. Edit license information in `LICENSE`.
4. Add your own `README.md`

### Developing

Project files are located in `src` directory. Coding style is automatically kept using `pre-commit`. Related configuration are stored in `pyproject.toml`.

### Testing

Add test scripts to `tests` directory. Each script should start with `test_`.

Execute `poetry run pytest` to perform unit test.
