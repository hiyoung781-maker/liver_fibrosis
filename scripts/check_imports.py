#!/usr/bin/env python3
"""Find third-party packages REINVENT4 imports at startup but does not declare.

REINVENT4 v4.5.11 under-declares its dependencies: `torchvision` and `scipy` are
both imported at module level on the path from `reinvent.Reinvent`, yet neither
appears in its pyproject.toml. Installing with --no-deps (which this project must
do, since three declared dependencies are deliberately omitted) turns that into a
crash at first run rather than at install time.

This walks the module-level import graph the way CPython would - following only
imports that execute at import time, and treating `from pkg import submodule` as
importing `pkg.submodule` - then reports third-party modules that are not in the
installed set.

    python3 scripts/check_imports.py [REINVENT4_dir] [entry.module]

Re-run it after changing the pinned REINVENT4 version.
"""

from __future__ import annotations

import ast
import pathlib
import sys

# What env/environment.yml + env/requirements-pip*.txt actually install, by the
# name you would `import`, plus the transitive pieces pip pulls in with torch.
INSTALLED = {
    "torch", "torchvision", "rdkit", "numpy", "scipy", "pandas", "matplotlib", "PIL",
    "pydantic", "dotenv", "yaml", "requests", "requests_mock", "tenacity", "tensorboard",
    "tomli", "tqdm", "typing_extensions", "xxhash", "funcy", "mmpdb", "molvs", "pytest",
    "tensorboardX", "setuptools", "pkg_resources", "jinja2", "filelock", "fsspec",
    "networkx", "sympy", "triton", "markupsafe", "mpmath",
}

# Deliberately absent; each backs one scoring component that the plugin importer
# skips on ImportError. See env/requirements-pip.txt for the reasoning.
OMITTED = {"chemprop", "descriptastorus", "openeye"}


def main(root: pathlib.Path, entry: str) -> int:
    stdlib = set(sys.stdlib_module_names)

    def path_of(mod: str) -> pathlib.Path | None:
        p = root / (mod.replace(".", "/") + ".py")
        if p.exists():
            return p
        p = root / mod.replace(".", "/") / "__init__.py"
        return p if p.exists() else None

    def toplevel(path: pathlib.Path):
        """Imports that run at module import time, including inside a top-level `if`."""
        out = []

        def handle(node):
            if isinstance(node, ast.Import):
                out.extend((a.name, 0, []) for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                out.append((node.module or "", node.level, [a.name for a in node.names]))

        for node in ast.parse(path.read_text(errors="replace")).body:
            handle(node)
            if isinstance(node, ast.If):           # e.g. `if SYSTEM != "Windows":`
                for sub in node.body:
                    handle(sub)
        return out

    seen: set[str] = set()
    external: dict[str, list[str]] = {}

    def walk(mod: str, chain: list[str]) -> None:
        if mod in seen:
            return
        seen.add(mod)
        path = path_of(mod)
        if path is None:
            return
        is_pkg = (root / mod.replace(".", "/") / "__init__.py").exists()
        pkg = mod if is_pkg else mod.rsplit(".", 1)[0]
        for name, level, imported in toplevel(path):
            base = (f"{pkg}.{name}" if name else pkg) if level else name
            top = base.split(".")[0]
            if top in ("reinvent", "reinvent_plugins"):
                walk(base, chain + [mod])
                # `from pkg import submodule` imports pkg.submodule too - missing this
                # is what hid scipy and torchvision on the first pass.
                for n in imported:
                    if path_of(f"{base}.{n}"):
                        walk(f"{base}.{n}", chain + [mod])
            elif top not in stdlib and top != "__future__":
                external.setdefault(top, chain + [mod])

    walk(entry, [])
    print(f"modules reached from {entry}: {len(seen)}\n")

    missing = []
    for name in sorted(external):
        if name in INSTALLED:
            continue
        if name in OMITTED:
            print(f"  {name:16s} omitted on purpose, but REACHED AT STARTUP - "
                  f"it must be plugin-only")
            missing.append(name)
            continue
        missing.append(name)
        print(f"  {name:16s} *** NOT INSTALLED ***")
        print(f"      via: {' -> '.join(external[name][-3:])}")

    if missing:
        print(f"\nMISSING: {missing}")
        return 1
    print("Every third-party module imported at startup is installed.")
    return 0


if __name__ == "__main__":
    base = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "REINVENT4")
    module = sys.argv[2] if len(sys.argv) > 2 else "reinvent.Reinvent"
    sys.exit(main(base, module))
