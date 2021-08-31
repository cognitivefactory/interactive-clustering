"""Generate the credits page."""

import functools
import re
from itertools import chain
from pathlib import Path

import mkdocs_gen_files
import toml
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment


def get_credits_data() -> dict:
    """Return data used to generate the credits file.

    Returns:
        Data required to render the credits template.
    """
    project_dir = Path(__file__).parent.parent
    metadata = toml.load(project_dir / "pyproject.toml")["project"]
    metadata_pdm = toml.load(project_dir / "pyproject.toml")["tool"]["pdm"]
    lock_data = toml.load(project_dir / "pdm.lock")
    project_name = metadata["name"]

    all_dependencies = chain(
        metadata.get("dependencies", []),
        chain(*metadata.get("optional-dependencies", {}).values()),
        chain(*metadata_pdm.get("dev-dependencies", {}).values()),
    )
    direct_dependencies = {re.sub(r"[^\w-].*$", "", dep) for dep in all_dependencies}
    direct_dependencies = {dep.lower() for dep in direct_dependencies}
    indirect_dependencies = {pkg["name"].lower() for pkg in lock_data["package"]}
    indirect_dependencies -= direct_dependencies

    return {
        "project_name": project_name,
        "direct_dependencies": sorted(direct_dependencies),
        "indirect_dependencies": sorted(indirect_dependencies),
        "more_credits": None,  # TODO: change https://github.com/pawamoy/jinja-templates/credits.md : `{%- if more_credits %}` -> `{%- if more_credits is defined %}`
    }


@functools.lru_cache(maxsize=None)
def get_credits():
    """Return credits as Markdown.

    Returns:
        The credits page Markdown.
    """
    jinja_env = SandboxedEnvironment(undefined=StrictUndefined)
    # commit = "c78c29caa345b6ace19494a98b1544253cbaf8c1"
    # template_url = f"https://raw.githubusercontent.com/pawamoy/jinja-templates/{commit}/credits.md"
    template_data = get_credits_data()
    # template_text = urlopen(template_url).read().decode("utf8")  # noqa: S310
    template_text = """
    <!--
        Template repository: https://github.com/pawamoy/jinja-templates
        Template path: credits.md
    -->

    # Credits

    These projects were used to build `{{ project_name }}`. **Thank you!**

    [`python`](https://www.python.org/) |
    [`pdm`](https://pdm.fming.dev/) |
    [`copier-pdm`](https://github.com/pawamoy/copier-pdm)

    ### Direct dependencies

    {% for dep in direct_dependencies -%}
    [`{{ dep }}`](https://pypi.org/project/{{ dep }}/){% if not loop.last %} |{% endif %}
    {%- endfor %}

    ### Indirect dependencies

    {% for dep in indirect_dependencies -%}
    [`{{ dep }}`](https://pypi.org/project/{{ dep }}/){% if not loop.last %} |{% endif %}
    {%- endfor %}
    {%- if more_credits %}

    **[More credits from the author]({{ more_credits }})**
    {%- endif %}
    """
    return jinja_env.from_string(template_text).render(**template_data)


with mkdocs_gen_files.open("credits.md", "w") as fd:
    fd.write(get_credits())
mkdocs_gen_files.set_edit_path("credits.md", "gen_credits.py")
