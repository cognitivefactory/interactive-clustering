# Contributing

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

## Environment setup

Nothing easier!
Follow the instructions below.

!!! note
    We **STRONGLY** recommend using a Linux distribution for Python development (Windows sometimes leads to obscure compatibility errors...)


1. Install [`Git`](https://git-scm.com/) to version and track our software changes.

    - On _Windows_, use the official installer: [`Git-for-Windows`](https://git-scm.com/download/win).

    - On _Linux_, simply use your package manager.

    !!! note
        `Git-for-Windows` doesn't provide the command `make`. In following step, use `pdm` instead.


1. Install [`Python`](https://www.python.org/) as programming language for this projet.

    - On _Windows_, use the official installer: [Python Releases for Windows](https://www.python.org/downloads/windows/).

    - On _Linux_, simply use your package manager.

    !!! note
        You can also use use [`pyenv`](https://github.com/pyenv/pyenv).

        ```bash
        # install pyenv
        git clone https://github.com/pyenv/pyenv ~/.pyenv

        # setup pyenv (you should also put these three lines in .bashrc or similar)
        export PATH="${HOME}/.pyenv/bin:${PATH}"
        export PYENV_ROOT="${HOME}/.pyenv"
        eval "$(pyenv init -)"

        # install Python 3.8
        pyenv install 3.8

        # make it available globally
        pyenv global system 3.8
        ```


1. Fork and clone the repository:

    ```bash
    git clone https://github.com/cognitivefactory/interactive-clustering/
    cd interactive-clustering
    ```


1. Install the dependencies of the projet with:

    ```bash
    cd interactive-clustering
    make setup # on Linux
    pdm install # on Windows
    ```

    !!! note
        If it fails for some reason (especially on Windows), you'll need to install [`pipx`](https://github.com/pypa/pipx) and [`pdm`](https://github.com/pdm-project/pdm) manually.

        You can install them with:

        ```bash
        python3 -m pip install --user pipx
        pipx install pdm
        ```

        Now you can try running `make setup` again, or simply `pdm install`.

Your project is now ready and dependencies are installed.


## Available template tasks

This project uses [duty](https://github.com/pawamoy/duty) to run tasks.
A Makefile is also provided.
To run a task, use `make TASK` on _Linux_ and `pdm run duty TASK` _on Windows_.

To show the available template task:

```bash
make help # on Linux
pdm run duty --list # on Windows
```

The Makefile will try to run certain tasks on multiple Python versions.
If for some reason you don't want to run the task on multiple Python versions, you can do one of the following:

1. `export PYTHON_VERSIONS= `: this will run the task
   with only the current Python version
2. run the task directly with `pdm run duty TASK`

The Makefile detects if a virtual environment is activated, so `make`/`pdm` will work the same with the virtualenv activated or not.

## Development journey

As usual:

1. create a new branch: `git checkout -b feature-or-bugfix-name`
1. edit the code and/or the documentation

If you updated the documentation or the project dependencies:

1. run `make docs-regen`
1. run `make docs-serve`, go to http://localhost:8000 and check that everything looks good

**Before committing:**

1. run `make format` to auto-format the code
1. run `make check` to check everything (fix any warning)
1. run `make test` to run the tests (fix any issue)
1. follow our [commit message convention](#commit-message-convention)

If you are unsure about how to fix or ignore a warning, just let the continuous integration fail, and we will help you during review.

Don't bother updating the changelog, we will take care of this.

## Commit message convention

Commits messages must follow the [Angular style](https://gist.github.com/stephenparish/9941e89d80e2bc58a153#format-of-the-commit-message):

```
<type>[(scope)]: Subject

[Body]
```

Scope and body are optional. Type can be:

- `build`: About packaging, building wheels, etc.
- `chore`: About packaging or repo/files management.
- `ci`: About Continuous Integration.
- `docs`: About documentation.
- `feat`: New feature.
- `fix`: Bug fix.
- `perf`: About performance.
- `refactor`: Changes which are not features nor bug fixes.
- `style`: A change in code style/format.
- `tests`: About tests.

**Subject (and body) must be valid Markdown.**
If you write a body, please add issues references at the end:

```
Body.

References: #10, #11.
Fixes #15.
```

## Pull requests guidelines

Link to any related issue in the Pull Request message.

During review, we recommend using fixups:

```bash
# SHA is the SHA of the commit you want to fix
git commit --fixup=SHA
```

Once all the changes are approved, you can squash your commits:

```bash
git rebase -i --autosquash master
```

And force-push:

```bash
git push -f
```

If this seems all too complicated, you can push or force-push each new commit, and we will squash them ourselves if needed, before merging.
