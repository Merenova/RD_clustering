"""Side-import of /home/hyunjin/circuit-tracer (new repo) so we can use its
NNSightReplacementModel without disturbing the in-tree TransformerLens code.

We add the new repo to sys.path under a *separate* top-level alias
(circuit_tracer_new) to avoid colliding with the in-tree circuit_tracer package.
"""
from __future__ import annotations

import importlib
import importlib.util
import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType

_NEW_REPO_ROOT = Path("/home/hyunjin/circuit-tracer")
_ALIAS = "circuit_tracer_new"
_ORIG_PKG = "circuit_tracer"

# Track whether we're currently loading a new-repo module (for reentrancy guard)
_loading_new_repo: set[str] = set()


class _NewRepoLoader:
    """Loader that loads a module from a file but registers it under two names."""

    def __init__(self, orig_name: str, alias_name: str, file_path: Path, is_pkg: bool):
        self._orig_name = orig_name
        self._alias_name = alias_name
        self._file_path = file_path
        self._is_pkg = is_pkg

    def create_module(self, spec):
        return None  # use default

    def exec_module(self, module):
        _loading_new_repo.add(self._orig_name)
        try:
            with open(self._file_path, "rb") as f:
                code = compile(f.read(), str(self._file_path), "exec")
            exec(code, module.__dict__)
        finally:
            _loading_new_repo.discard(self._orig_name)


class _NewRepoFinder:
    """Meta path finder: when a new-repo module does 'from circuit_tracer.xxx import ...',
    redirect to load from the new repo and register under circuit_tracer_new.xxx.
    """

    def find_spec(self, fullname, path, target=None):
        # Only intercept circuit_tracer.* imports
        if not fullname.startswith(_ORIG_PKG + "."):
            return None

        # Only intercept when we're in the middle of loading a new-repo module
        if not _loading_new_repo:
            return None

        # Don't re-intercept modules we already have from in-tree (that are ok)
        if fullname in sys.modules:
            return None

        # Map circuit_tracer.foo.bar -> circuit_tracer_new.foo.bar
        alias_name = _ALIAS + fullname[len(_ORIG_PKG):]
        sub_rel = fullname[len(_ORIG_PKG) + 1:].replace(".", "/")

        candidates = [
            _NEW_REPO_ROOT / _ORIG_PKG / (sub_rel + ".py"),
            _NEW_REPO_ROOT / _ORIG_PKG / sub_rel / "__init__.py",
        ]

        file_path = None
        is_pkg = False
        for candidate in candidates:
            if candidate.exists():
                file_path = candidate
                is_pkg = candidate.name == "__init__.py"
                break

        if file_path is None:
            return None  # fall through to normal finders

        loader = _NewRepoLoader(fullname, alias_name, file_path, is_pkg)
        submodule_search = [str(file_path.parent)] if is_pkg else None

        spec = importlib.machinery.ModuleSpec(
            fullname,
            loader,
            origin=str(file_path),
            is_package=is_pkg,
        )
        if submodule_search is not None:
            spec.submodule_search_locations = submodule_search

        return spec


def _install_finder():
    if not any(isinstance(f, _NewRepoFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _NewRepoFinder())


def _load_new_circuit_tracer():
    if _ALIAS in sys.modules:
        return sys.modules[_ALIAS]
    if not _NEW_REPO_ROOT.exists():
        raise RuntimeError(f"New circuit-tracer not found at {_NEW_REPO_ROOT}")

    _install_finder()

    spec = importlib.util.spec_from_file_location(
        _ALIAS,
        _NEW_REPO_ROOT / _ORIG_PKG / "__init__.py",
        submodule_search_locations=[str(_NEW_REPO_ROOT / _ORIG_PKG)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_ALIAS] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_new_repo_submodule(alias_name: str) -> ModuleType:
    """Load a submodule from the new repo under the alias namespace."""
    if alias_name in sys.modules:
        return sys.modules[alias_name]

    _load_new_circuit_tracer()  # ensure top-level is loaded

    sub_rel = alias_name[len(_ALIAS) + 1:].replace(".", "/")
    candidates = [
        _NEW_REPO_ROOT / _ORIG_PKG / (sub_rel + ".py"),
        _NEW_REPO_ROOT / _ORIG_PKG / sub_rel / "__init__.py",
    ]

    file_path = None
    is_pkg = False
    for candidate in candidates:
        if candidate.exists():
            file_path = candidate
            is_pkg = candidate.name == "__init__.py"
            break

    if file_path is None:
        raise ModuleNotFoundError(f"Cannot find {alias_name} in {_NEW_REPO_ROOT}")

    # Ensure parent package is registered in sys.modules
    parts = alias_name.split(".")
    for i in range(2, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            _load_new_repo_submodule(parent)

    orig_name = _ORIG_PKG + alias_name[len(_ALIAS):]

    loader = _NewRepoLoader(orig_name, alias_name, file_path, is_pkg)
    spec = importlib.machinery.ModuleSpec(
        alias_name,
        loader,
        origin=str(file_path),
        is_package=is_pkg,
    )
    if is_pkg:
        spec.submodule_search_locations = [str(file_path.parent)]

    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = ".".join(parts[:-1]) if not is_pkg else alias_name
    mod.__name__ = alias_name
    sys.modules[alias_name] = mod

    # While exec'ing, mark the orig name active so _NewRepoFinder intercepts
    # any circuit_tracer.* sub-imports that come from this module
    _loading_new_repo.add(orig_name)
    try:
        loader.exec_module(mod)
    finally:
        _loading_new_repo.discard(orig_name)

    return mod


def get_nnsight_replacement_model_cls():
    _install_finder()
    _load_new_circuit_tracer()
    mod = _load_new_repo_submodule(f"{_ALIAS}.replacement_model")
    mod = _load_new_repo_submodule(f"{_ALIAS}.replacement_model.replacement_model_nnsight")
    return mod.NNSightReplacementModel


def get_nnsight_attribution_context_cls():
    _install_finder()
    _load_new_circuit_tracer()
    mod = _load_new_repo_submodule(f"{_ALIAS}.attribution")
    mod = _load_new_repo_submodule(f"{_ALIAS}.attribution.context_nnsight")
    return mod.AttributionContext
