# spatial_tool/support.py
"""Module containing support tools"""


# pylint: disable=protected-access


import inspect
import importlib
from typing import get_origin, get_args, Annotated, Any, Literal

from pydantic.fields import FieldInfo
import numpy as np


def _get_field_description(meta):

    desc = getattr(meta, "description", None)
    if desc is not None:
        return desc
    fi = getattr(meta, "field_info", None)
    return getattr(fi, "description", None) if fi is not None else None


def _doc_function_fields(fn):

    sig = inspect.signature(fn)
    docs = []

    # document parameters
    for name, param in sig.parameters.items():
        ann = param.annotation
        default = param.default
        typ = ann
        desc = None

        # handle Annotated types for parameters
        if get_origin(ann) is Annotated:
            args = get_args(ann)
            typ = args[0]
            for meta in args[1:]:
                if isinstance(meta, FieldInfo):
                    desc = _get_field_description(meta)
                    break

        # prepare type string
        try:
            type_str = typ.__name__
        except Exception:
            type_str = str(typ)

        # prepare default string
        default_str = default if default is not inspect._empty else None

        # build param doc line
        if desc:
            line = f":param {name} ({type_str}): {desc}"
            if default_str is not None:
                line += f" (default={default_str})"
            docs.append(line)
        else:
            line = f":param {name} ({type_str}):"
            if default_str is not None:
                line += f" (default={default_str})"
            docs.append(line)

    # document return annotation
    ret_ann = sig.return_annotation
    ret_desc = None
    ret_typ = None
    if get_origin(ret_ann) is Annotated:
        ret_args = get_args(ret_ann)
        ret_typ = ret_args[0]
        for meta in ret_args[1:]:
            if isinstance(meta, FieldInfo):
                ret_desc = _get_field_description(meta)
                break
    elif ret_ann is not inspect._empty and ret_ann is not None:
        # non-Annotated return type
        ret_typ = ret_ann

    if ret_typ:

        try:
            ret_type_str = ret_typ.__name__
        except Exception:
            ret_type_str = str(ret_typ)

        if ret_desc:
            docs.append(f":returns ({ret_type_str}): {ret_desc}")
        else:
            docs.append(f":returns ({ret_type_str}):")

    # combine with existing docstring
    existing = fn.__doc__.rstrip() if fn.__doc__ else ""
    generated = "\n".join(docs)
    fn.__doc__ = f"{existing}\n\n{generated}".strip()

    return fn


def _doc_class_fields(cls):

    lines = []
    annotations = {}

    # collect annotations from the class and all base classes
    for base in reversed(cls.__mro__):
        annotations.update(getattr(base, '__annotations__', {}))

    for name, ann in annotations.items():
        default = getattr(cls, name, inspect._empty)
        typ = ann
        desc = None

        # handle Annotated types for attributes
        if get_origin(ann) is Annotated:
            args = get_args(ann)
            typ = args[0]
            for meta in args[1:]:
                if isinstance(meta, FieldInfo):
                    desc = _get_field_description(meta)
                    break

        # prepare type string
        try:
            type_str = typ.__name__
        except Exception:
            type_str = str(typ)

        # prepare default string
        default_str = default if default is not inspect._empty else None

        # build attribute doc line
        if desc:
            line = f":attr {name} ({type_str}): {desc}"
            if default_str is not None:
                line += f"  (default={default_str})"
            lines.append(line)
        else:
            line = f":attr {name} ({type_str}):)"
            if default_str is not None:
                line += f" (default={default_str}"
            lines.append(line)

    # combine with existing docstring
    existing = cls.__doc__.rstrip() if cls.__doc__ else ""
    generated = "\n".join(lines)
    cls.__doc__ = f"{existing}\n\n{generated}".strip()

    return cls


def _get_backend(backend: Literal['default', 'cupy']) -> Any:

    if backend == 'cupy':
        try:
            return importlib.import_module('cupy')
        except ImportError as e:
            msg = """Backend 'cupy' requested but CuPy is not installed.
            Install with:
                pip install cupy-cuda12x   # NVIDIA CUDA 12.x
                pip install cupy-cuda11x   # NVIDIA CUDA 11.x"""
            raise ImportError(msg) from e
    else:
        return np
