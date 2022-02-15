#!/usr/bin/env bash
set -e

# un/comment this if you want to enable/disable multiple Python versions
#PYTHON_VERSIONS="${PYTHON_VERSIONS-3.7 3.8 3.9 3.10}"

install_with_pipx() {
    if ! command -v "$1" &>/dev/null; then
        if ! command -v pipx &>/dev/null; then
            python3 -m pip install --user pipx
        fi
        pipx install "$1"
    fi
}

install_with_pipx pdm

if [ -n "${PYTHON_VERSIONS}" ]; then
    for python_version in ${PYTHON_VERSIONS}; do
        if pdm use -f "${python_version}" &>/dev/null; then  # python${python_version}
            echo "> Using Python ${python_version} interpreter"
            pdm install
        else
            echo "> pdm use -f ${python_version}: Python interpreter not available?" >&2  # python${python_version}
        fi
    done
else
    pdm install
fi
