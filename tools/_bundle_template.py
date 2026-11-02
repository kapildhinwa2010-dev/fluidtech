#!/usr/bin/env python3
"""Generated single-file entry point for the Superfluid package."""

import os
import sys
import types

_SOURCES = r'''
%%SOURCES%%
'''


def _build_package():
    package = types.ModuleType("superfluid")
    package.__path__ = []
    package.__package__ = "superfluid"
    sys.modules["superfluid"] = package
    blocks = {}
    current = None
    for line in _SOURCES.splitlines():
        if line.startswith("#--- FILE: "):
            current = line[len("#--- FILE: "):].split(" ---")[0].strip()
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)

    module_paths = [path for path in blocks if path.endswith(".py")]
    module_paths.sort(key=lambda path: path.endswith("/__init__.py"))
    for path in module_paths:
        if path == "selftest.py":
            continue
        if path.endswith("/__init__.py"):
            exec(compile("\n".join(blocks[path]), path, "exec"), package.__dict__)
            continue
        name = path.rsplit("/", 1)[-1][:-3]
        module = types.ModuleType(f"superfluid.{name}")
        module.__package__ = "superfluid"
        module.__file__ = path
        sys.modules[module.__name__] = module
        exec(compile("\n".join(blocks[path]), path, "exec"), module.__dict__)
        setattr(package, name, module)
    return package, blocks


_package, _blocks = _build_package()
globals().update({name: getattr(_package, name) for name in _package.__all__})

__version__ = _package.__version__


def _run_selftest():
    source = "\n".join(_blocks["selftest.py"])
    namespace = {"__name__": "__main__", "__file__": "selftest.py"}
    sys.path.insert(0, os.getcwd())
    try:
        exec(compile(source, "selftest.py", "exec"), namespace)
    except SystemExit as error:
        return int(error.code or 0)
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_run_selftest())
