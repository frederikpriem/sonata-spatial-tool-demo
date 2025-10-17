# spatial_tool/function.py
"""Module containing the LayerFunction, ObjectiveFunction and DecisionFunction class definitions"""


import importlib
from typing import Callable, Iterable, Optional, Union, List, Tuple, Dict, Annotated

from pydantic import BaseModel, Field
import numpy as np

from .support import _doc_class_fields


def _function_from_path(path: str) -> Callable:

    mod_name, func_name = path.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    
    return getattr(mod, func_name)


class _BaseFunction(BaseModel):

    input_layer_ids: Annotated[Union[List[str], Tuple[str]], Field(
        min_length=1,
        description="""IDs of the layers that are to be passed as inputs to the function"""
    )]
    kwargs: Annotated[Optional[Dict], Field(
        description="""Dictionary of keyword arguments to be passed to the function"""
    )] = {}

    def model_post_init(self, __context) -> None:  # pylint: disable=arguments-differ

        super().model_post_init(__context)

        if isinstance(self.function, str):
            try:
                function = _function_from_path(self.function)
            except:
                msg = f"""Function could not be imported from given import path '{self.function}'"""
                raise ValueError(msg)

            if not callable(function):
                msg = f"""The object returned by given import path '{self.function}' is not
                    callable"""
                raise ValueError(msg)

    def _get_function(self):

        if isinstance(self.function, str):
            function = _function_from_path(self.function)
        else:
            function = self.function

        return function


@_doc_class_fields
class LayerFunction(_BaseFunction):

    """
    _summary_
    """

    function: Annotated[Union[Callable[..., np.ndarray], str], Field(
        description="""Function that takes input layers and returns a new layer as output. Can
            either be a callable or a string representing the import path to the function
            (e.g. 'module.submodule.function_name')"""
    )]

    def _evaluate(self, layers: Iterable[np.ndarray]) -> np.ndarray:

        function = self._get_function()
        return function(layers, **self.kwargs)


@_doc_class_fields
class DecisionFunction(_BaseFunction):

    """
    _summary_
    """

    function: Annotated[Union[Callable[..., np.ndarray], str], Field(
        description="""Function that takes input layers and coefficients, and returns a decision
            layer as output. Can either be a callable or a string representing the import path to
            the function (e.g. 'module.submodule.function_name')."""
    )]
    num_coef: Annotated[int, Field(
        ge=1,
        description="""Number of coefficients that the decision function expects, must be at least
            1."""
    )]

    def _evaluate(self, layers: Iterable[np.ndarray], coefficients: Iterable[float]) -> np.ndarray:

        function = self._get_function()
        return function(layers, coefficients, **self.kwargs)


@_doc_class_fields
class ObjectiveFunction(_BaseFunction):

    """
    _summary_
    """

    function: Annotated[Union[Callable[..., Union[bool, int, float]], str], Field(
        description="""Function that takes input layers and returns a single numeric value as output
        . Can either be a callable or a string representing the import path to the function (e.g.
        'module.submodule.function_name')."""
    )]

    def _evaluate(self, layers: Iterable[np.ndarray]) -> Union[bool, int, float]:

        function = self._get_function()
        return function(layers, **self.kwargs)
