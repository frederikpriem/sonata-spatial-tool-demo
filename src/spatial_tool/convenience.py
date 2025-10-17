# spatial_tool/convenience.py
"""Module containing convenience functions for layer creation"""


from typing import Any, Iterable, Optional, List, Union, Literal, Annotated, Tuple, TYPE_CHECKING
import importlib

import pydantic
from pydantic import Field, ConfigDict
import numpy as np

from .support import _doc_function_fields, _get_backend


try:
    import cupy as cp
except ImportError:
    cp = None


try:
    import cupy as cp
    NDArrayType = Union[np.ndarray, cp.ndarray]
except ImportError:
    cp = None
    NDArrayType = np.ndarray  # fallback


if TYPE_CHECKING:
    import cupy as cp


connectivity_map = {
    1: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    2: np.ones((3, 3), int)
}

validate_all_types = pydantic.validate_arguments(
    config=ConfigDict(arbitrary_types_allowed=True)
)





@_doc_function_fields
@validate_all_types
def indicate(
    input_layers: Annotated[List[NDArrayType], Field(...,
        description="""List containing exactly one input layer array."""
    )],
    target: Annotated[Union[float, int, bool], Field(...,
        description="""Target value with which the indicator function is to be evaluated."""
    )],
    predicate: Annotated[Optional[Literal['eq', 'ne', 'lt', 'le', 'gt', 'ge']], Field(...,
        description="""Predicate with which the indicator function is to be evaluated."""
    )] = 'eq',
    value_if_true: Annotated[Optional[bool], Field(
        description="""Value to return if predicate is True."""
    )] = True,
    value_if_false: Annotated[Optional[bool], Field(
        description="""Value to return if predicate is False."""
    )] = False,
    backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Backend used to evaluate the indicator function."""
    )] = 'default',
    return_np_array: Annotated[Optional[bool], Field(
        description="""Whether to return a numpy array. Only has effect if backend='cupy'."""
    )] = True
) -> NDArrayType:

    """
    _summary_
    """

    input_layer = input_layers[0]
    xp = _get_backend(backend)
    input_layer = xp.asarray(input_layer)

    ind = None

    if predicate == 'eq':
        ind = xp.where(input_layer == target, value_if_true, value_if_false)
    elif predicate == 'ne':
        ind = xp.where(input_layer != target, value_if_true, value_if_false)
    elif predicate == 'lt':
        ind = xp.where(input_layer < target, value_if_true, value_if_false)
    elif predicate == 'le':
        ind = xp.where(input_layer <= target, value_if_true, value_if_false)
    elif predicate == 'gt':
        ind = xp.where(input_layer > target, value_if_true, value_if_false)
    elif predicate == 'ge':
        ind = xp.where(input_layer >= target, value_if_true, value_if_false)

    if backend == 'cupy' and return_np_array:
        ind = ind.get()

    return ind


@_doc_function_fields
@validate_all_types
def add(
    input_layers: Annotated[List[NDArrayType], Field(...,
        description="""List containing exactly one input layer array."""
    )],
    term: Annotated[float, Field(...,
        description="""Term to be added to the input layer."""
    )],
    backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Backend used to perform the addition."""
    )] = 'default',
    return_np_array: Annotated[Optional[bool], Field(
        description="""Whether to return a numpy array. Only has effect if backend='cupy'."""
    )] = True
) -> NDArrayType:

    """
    _summary_
    """

    input_layer = input_layers[0]
    xp = _get_backend(backend)
    input_layer = xp.asarray(input_layer)

    ad = input_layer + term

    if backend == 'cupy' and return_np_array:
        ad = ad.get()

    return ad


@_doc_function_fields
@validate_all_types
def multiply(
    input_layers: Annotated[List[NDArrayType], Field(...,
        description="""List containing exactly one input layer array."""
    )],
    factor: Annotated[float, Field(
        description="""Factor with which the input layer is to be multiplied."""
    )],
    backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Backend used to perform the multiplication."""
    )] = 'default',
    return_np_array: Annotated[Optional[bool], Field(
        description="""Whether to return a numpy array. Only has effect if backend='cupy'."""
    )] = True
) -> NDArrayType:

    """
    _summary_
    """

    input_layer = input_layers[0]
    xp = _get_backend(backend)
    input_layer = xp.asarray(input_layer)

    mult = input_layer * factor

    if backend == 'cupy' and return_np_array:
        mult = mult.get()

    return mult


@_doc_function_fields
@validate_all_types
def power(
    input_layers: Annotated[List[NDArrayType], Field(...,
        description="""List containing exactly one input layer array."""
    )],
    exponent: Annotated[float, Field(...,
        description="""Exponent with which the input layer is to be power-transformed."""
    )],
    backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Backend used to perform the power transform."""
    )] = 'default',
    return_np_array: Annotated[Optional[bool], Field(
        description="""Whether to return a numpy array. Only has effect if backend='cupy'."""
    )] = True
) -> NDArrayType:

    """
    _summary_
    """

    input_layer = input_layers[0]
    xp = _get_backend(backend)
    input_layer = xp.asarray(input_layer)

    pwr = input_layer ** exponent

    if backend == 'cupy' and return_np_array:
        pwr = pwr.get()

    return pwr


@_doc_function_fields
@validate_all_types
def aggregate(
    input_layers: Annotated[List[NDArrayType], Field(...,
        description="""List containing at least two input layer arrays."""
    )],
    operator: Annotated[
        Literal['sum', 'product', 'mean', 'std', 'min', 'max', 'and', 'or'],
        Field(
            ...,
            description="""Operator with which the aggregation function is to be evaluated."""
        )
    ],
    backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Backend used to perform the aggregation."""
    )] = 'default',
    return_np_array: Annotated[Optional[bool], Field(
        description="""Whether to return a numpy array. Only has effect if backend='cupy'."""
    )] = True
) -> NDArrayType:

    """
    _summary_
    """

    xp = _get_backend(backend)
    input_layers = [xp.asarray(input_layer) for input_layer in input_layers]

    input_layers = [xp.expand_dims(input_layer, axis=2) for input_layer in input_layers]
    aggr = xp.concatenate(input_layers, axis=2)

    if operator == 'sum':
        aggr = aggr.sum(axis=2)
    elif operator == 'product':
        aggr = aggr.prod(axis=2)
    elif operator == 'mean':
        aggr = aggr.mean(axis=2)
    elif operator == 'std':
        aggr = aggr.std(axis=2)
    elif operator == 'min':
        aggr = aggr.min(axis=2)
    elif operator == 'max':
        aggr = aggr.max(axis=2)
    elif operator == 'and':
        aggr = np.all(aggr, axis=2)
    elif operator == 'or':
        aggr = np.any(aggr, axis=2)

    aggr = aggr.squeeze()

    if backend == 'cupy' and return_np_array:
        aggr = aggr.get()

    return aggr


@_doc_function_fields
@validate_all_types
def combine(
    input_layers: Annotated[List[NDArrayType], Field(...,
        description="""List of input layer arrays."""
    )],
    weights: Annotated[List[Union[float, int, bool]], Field(...,
        description="""Weights used to sum the input layer arrays. Must have same length as
            input_layers."""
    )],
    bias: Annotated[Optional[Union[float, int, bool]], Field(
        description="""Optional bias to be added to linear combination."""
    )] = 0.,
    backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Backend used to perform the combine operation."""
    )] = 'default',
    return_np_array: Annotated[Optional[bool], Field(
        description="""Whether to return a numpy array. Only has effect if backend='cupy'."""
    )] = True
) -> NDArrayType:

    """
    _summary_
    """

    xp = _get_backend(backend)
    weights = list(weights)

    assert len(input_layers) == len(weights), """input_layers and weights must have same length"""

    input_layers = [xp.asarray(input_layer) for input_layer in input_layers]

    comb = xp.zeros(input_layers[0].shape, dtype=float)

    for input_layer, weight in zip(input_layers, weights):
        comb += input_layer * weight

    comb += bias

    if backend == 'cupy' and return_np_array:
        comb = comb.get()

    return comb


@_doc_function_fields
@validate_all_types
def logistic(
    input_layers: Annotated[List[NDArrayType], Field(...,
        description="""List of input layer arrays, in this case containing exactly one array."""
    )],
    l: Annotated[Optional[Union[float, int]], Field(
        description="""Carrying capacity of the logistic function."""
    )] = 1,
    k: Annotated[Optional[Union[float, int]], Field(
        description="""Logistic growth rate of the logistic function."""
    )] = 1,
    x_zero: Annotated[Optional[Union[float, int]], Field(
        description="""Midpoint of the logistic function."""
    )] = 0,
    backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Backend used to perform the logistic transform."""
    )] = 'default',
    return_np_array: Annotated[Optional[bool], Field(
        description="""Whether to return a numpy array. Only has effect if backend='cupy'."""
    )] = True
) -> NDArrayType:

    """
    _summary_
    """

    input_layer = input_layers[0]
    xp = _get_backend(backend)
    input_layer = xp.asarray(input_layer)

    logist = l / (1 + xp.exp(-k * (input_layer - x_zero)))

    if backend == 'cupy' and return_np_array:
        logist = logist.get()

    return logist


@_doc_function_fields
@validate_all_types
def logit(
    input_layers: Annotated[List[NDArrayType], Field(...,
        description="""List of input layer arrays, in this case containing exactly one array. All
            array values must be within [0, 1] range."""
    )],
    backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Backend used to perform the logit transform."""
    )] = 'default',
    return_np_array: Annotated[Optional[bool], Field(
        description="""Whether to return a numpy array. Only has effect if backend='cupy'."""
    )] = True
) -> NDArrayType:

    """
    _summary_
    """

    input_layer = input_layers[0]
    xp = _get_backend(backend)
    input_layer = xp.asarray(input_layer)

    lgt = xp.log(input_layer / (1 - input_layer))

    if backend == 'cupy' and return_np_array:
        lgt = lgt.get()

    return lgt


@_doc_function_fields
@validate_all_types
def standardize(
    input_layers: Annotated[List[NDArrayType], Field(...,
        description="""List containing exactly one input layer array."""
    )],
    backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Backend used to perform the normalization."""
    )] = 'default',
    return_np_array: Annotated[Optional[bool], Field(
        description="""Whether to return a numpy array. Only has effect if backend='cupy'."""
    )] = True
) -> NDArrayType:

    """
    _summary_
    """

    input_layer = input_layers[0]
    xp = _get_backend(backend)
    input_layer = xp.asarray(input_layer)

    mean = input_layer.mean()
    std = input_layer.std()

    stand = (input_layer - mean) / std

    if backend == 'cupy' and return_np_array:
        stand = stand.get()

    return stand


@_doc_function_fields
@validate_all_types
def normalize(
    input_layers: Annotated[List[NDArrayType], Field(...,
        description="""List containing exactly one input layer array."""
    )],
    backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Backend used to perform the normalization."""
    )] = 'default',
    return_np_array: Annotated[Optional[bool], Field(
        description="""Whether to return a numpy array. Only has effect if backend='cupy'."""
    )] = True
) -> NDArrayType:

    """
    _summary_
    """

    input_layer = input_layers[0]
    xp = _get_backend(backend)
    input_layer = xp.asarray(input_layer)

    norm = input_layer / input_layer.sum()

    if backend == 'cupy' and return_np_array:
        norm = norm.get()

    return norm


@_doc_function_fields
@validate_all_types
def minmax(
    input_layers: Annotated[List[NDArrayType], Field(...,
        description="""List containing exactly one input layer array."""
    )],
    backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Backend used to perform the minmax-transform."""
    )] = 'default',
    return_np_array: Annotated[Optional[bool], Field(
        description="""Whether to return a numpy array. Only has effect if backend='cupy'."""
    )] = True
) -> NDArrayType:

    """
    _summary_
    """

    input_layer = input_layers[0]
    xp = _get_backend(backend)
    input_layer = xp.asarray(input_layer)

    maxval = input_layer.max()
    minval = input_layer.min()

    mm = (input_layer - minval) / (maxval - minval)

    if backend == 'cupy' and return_np_array:
        mm = mm.get()

    return mm


@_doc_function_fields
@validate_all_types
def dilate(
    input_layers: Annotated[List[NDArrayType], Field(...,
        description="""List containing exactly one input layer array. The provided array must be
            (convertible to) binary."""
    )],
    iterations: Annotated[Optional[int], Field(
        ge=1,
        description="""Number of times that dilation must be performed consecutively.
            """
    )] = 1,
    connectivity: Annotated[Optional[int], Field(
        ge=1,
        le=2,
        description="""Connectivity to use for dilation, 1 = 4-neighborhood, 2 = 8-neigborhood."""
    )] = 1,
    backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Backend used to perform the dilate operation."""
    )] = 'default',
    return_np_array: Annotated[Optional[bool], Field(
        description="""Whether to return a numpy array. Only has effect if backend='cupy'."""
    )] = True
) -> NDArrayType:

    """
    _summary_
    """

    xp = _get_backend(backend)

    if backend == 'default':
        sp = importlib.import_module('scipy.ndimage')
    else:
        sp = importlib.import_module('cupyx.scipy.ndimage')

    binary = input_layers[0].astype(bool)
    structure = connectivity_map[connectivity]
    binary = xp.asarray(binary)
    structure = xp.asarray(structure)

    dil = sp.binary_dilation(binary,
        iterations=iterations,
        structure=structure
    )

    if backend == 'cupy' and return_np_array:
        dil = dil.get()

    return dil


@_doc_function_fields
@validate_all_types
def erode(
    input_layers: Annotated[List[NDArrayType], Field(...,
        description="""List containing exactly one input layer array. The provided array must be
            (convertible to) binary."""
    )],
    iterations: Annotated[Optional[int], Field(
        ge=1,
        description="""Number of times that erosion must be performed consecutively."""
    )] = 1,
    connectivity: Annotated[Optional[int], Field(
        ge=1,
        le=2,
        description="""Connectivity to use for erosion, 1 = 4-neighborhood, 2 = 8-neigborhood."""
    )] = 1,
    backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Backend used to perform the erode operation."""
    )] = 'default',
    return_np_array: Annotated[Optional[bool], Field(
        description="""Whether to return a numpy array. Only has effect if backend='cupy'."""
    )] = True
) -> NDArrayType:

    """
    _summary_
    """

    xp = _get_backend(backend)

    if backend == 'default':
        sp = importlib.import_module('scipy.ndimage')
    else:
        sp = importlib.import_module('cupyx.scipy.ndimage')

    binary = input_layers[0].astype(bool)
    structure = connectivity_map[connectivity]
    binary = xp.asarray(binary)
    structure = xp.asarray(structure)

    er = sp.binary_erosion(binary,
        iterations=iterations,
        structure=structure
    )

    if backend == 'cupy' and return_np_array:
        er = er.get()

    return er


@_doc_function_fields
@validate_all_types
def patch_size(
    input_layers: Annotated[List[NDArrayType], Field(...,
        description="""List containing exactly one input layer array. The provided array must be
            (convertible to) binary."""
    )],
    connectivity: Annotated[Optional[int], Field(
        ge=1,
        le=2,
        description="""Connectivity to use for patch identification, 1 = 4-neighborhood, 2 =
            8-neigborhood."""
    )] = 1,
    backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Backend used to perform the patch size operation."""
    )] = 'default',
    return_np_array: Annotated[Optional[bool], Field(
        description="""Whether to return a numpy array. Only has effect if backend='cupy'."""
    )] = True
) -> NDArrayType:

    """
    _summary_
    """

    xp = _get_backend(backend)

    if backend == 'default':
        sp = importlib.import_module('scipy.ndimage')
    else:
        sp = importlib.import_module('cupyx.scipy.ndimage')

    binary = input_layers[0].astype(bool)
    structure = connectivity_map[connectivity]
    binary = xp.asarray(binary)
    structure = xp.asarray(structure)

    labels, _ = sp.label(binary, structure=structure)
    counts = xp.bincount(labels.ravel())
    ps = counts[labels]
    ps[xp.logical_not(binary)] = 0

    if backend == 'cupy' and return_np_array:
        ps = ps.get()

    return ps


@_doc_function_fields
@validate_all_types
def convolve(
    input_layers: Annotated[List[NDArrayType], Field(...,
        description="""List containing exactly one input layer array."""
    )],
    radius_m : Annotated[float, Field(...,
        gt=0,
        description="""Outer radius, in meter, of the convolution kernel."""
    )],
    cell_size_m: Annotated[float, Field(...,
        gt=0,
        description="""Size of the cells in the model space, expressed in meter. Square cells are
            assumed."""
    )],
    radius_inner_m: Annotated[Optional[float], Field(
        ge=0,
        description="""Inner radius, in meter, of the convolution kernel. If
            radius_inner_m is greater than zero, the kernel will be ring-shaped instead of circular.
            """
    )] = 0,
    decay: Annotated[Optional[float], Field(
        ge=0.,
        description="""Parameter used to apply exponential distance decay on the convolution kernel.
            Defaults to zero (no distance decay). Only non-negative values are accepted. The larger
            the parameter value, the stronger the distance decay will be."""
    )] = 0.,
    mode: Annotated[Optional[Literal['reflect', 'constant', 'nearest', 'mirror', 'wrap']], Field(
        description="""How the array is extended beyond its edges for the purpose of convolution.
            See scipy.ndimage.convolve documentation for more info."""
    )] = 'constant',
    cval: Annotated[Optional[Union[int, float]], Field(
        description="""Value to use if mode = 'constant'."""
    )] = 0.,
    backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Backend used to perform the convolve operation."""
    )] = 'default',
    return_np_array: Annotated[Optional[bool], Field(
        description="""Whether to return a numpy array. Only has effect if backend='cupy'."""
    )] = True
) -> NDArrayType:

    """
    _summary_
    """

    xp = _get_backend(backend)

    if backend == 'default':
        sp = importlib.import_module('scipy.ndimage')
    else:
        sp = importlib.import_module('cupyx.scipy.ndimage')

    input_layer = input_layers[0]
    input_layer = xp.asarray(input_layer)

    radius_cells = int(xp.ceil(radius_m / cell_size_m))
    y, x = xp.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
    d = xp.sqrt(x**2 + y**2) * cell_size_m

    kernel = xp.exp(-1. * decay * d)
    kernel[d < radius_inner_m] = 0
    kernel[d > radius_m] = 0
    kernel /= kernel.sum()

    conv = sp.convolve(input_layer, kernel,
        mode=mode,
        cval=cval)

    if backend == 'cupy' and return_np_array:
        conv = conv.get()

    return conv
