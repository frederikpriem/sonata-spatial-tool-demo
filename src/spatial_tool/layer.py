# spatial_tool/layer.py
"""Module containing Layer class definition"""


# pylint: disable=protected-access


import copy
import os
from pathlib import Path
from typing import Optional, Union, Dict, Literal, Annotated, List

import numpy as np
import rasterio as rio
import pydantic
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict

from .function import LayerFunction
from .support import _doc_class_fields, _doc_function_fields


validate_all_types = pydantic.validate_arguments(
    config=ConfigDict(arbitrary_types_allowed=True)
)


@_doc_class_fields
class Layer(BaseModel):

    """
    _summary_
    """

    layer_id: Annotated[str, Field(
        description="""Unique identifier for the layer."""
    )]
    source: Annotated[Union[str, Path, LayerFunction, Literal['memory']], Field(
        description="""Source of the layer. Can either be a filepath string, a Path object, a
            LayerFunction instance, or "memory"."""
    )]
    scale: Annotated[Optional[Union[float, None]], Field(
        gt=0,
        description="""Scale factor by which the values of the layer will be divided when loaded
            from source file"""
    )] = None
    model_space: Annotated[Optional[bool], Field(
        description="""Whether the layer represents the model space layer. Can only be True for one
            layer."""
    )] = False
    label_map: Annotated[Optional[Union[Dict[str, int], None]], Field(
        description="""Dictionary containing a mapping of cover type labels to the corresponding
            raster values of the layer. Must be specified if model_space = True."""
    )] = None
    profile: Annotated[Optional[Dict], Field(
        description="""Dictionary containing the parameters needed by rasterio to write layers to
            GIS-supported file format. Note that if source is a file path, profile will be taken
            from the source file when the layer is loaded."""
    )] = {}
    cache: Annotated[Optional[bool], Field(
        description="""Whether to cache the layer values in memory. Cached values can quickly be
            reused when the layer is loaded or updated."""
    )] = False
    _values: Annotated[Union[np.ndarray, None], PrivateAttr()] = None
    _cached_values: Annotated[Union[np.ndarray, None], PrivateAttr()] = None

    def model_post_init(self, __context) -> None:  # pylint: disable=arguments-differ

        """
        Included only for pydantic compatibility. Don't use.
        """

        super().model_post_init(__context)

        if self.model_space and self.label_map is None:
            raise ValueError('Model space layer must have a label map defined')

        con1 = isinstance(self.source, Path)
        con2 = isinstance(self.source, str)
        if con1 or con2 and self.source != 'memory':
            if not os.path.exists(str(self.source)):
                raise ValueError(f'Source file {str(self.source)} not found')

    def _get_data_mask(self) -> Union[None, np.ndarray]:

        data_mask = None

        if self._values is not None:

            if 'nodata' in self.profile.keys():
                data_mask = np.where(self._values == self.profile['nodata'], False, True)
            else:
                data_mask = np.ones(self._values.shape).astype(bool)

        return data_mask

    @_doc_function_fields
    def load(self,
        from_cache: Annotated[Optional[bool], Field(
            description="""Whether to load layer values from cache. Only has effect if cache=True
                and if there are cached values."""
        )] = False,
        to_cache: Annotated[Optional[bool], Field(
            description="""Whether to store a copy of the loaded values in cache. Only has effect if
                cache=True."""
        )] = False
    ) -> None:

        """
        _summary_
        """

        con1 = isinstance(self.source, str) or isinstance(self.source, Path)

        if con1 and from_cache and self.cache and self._cached_values is not None:

            self._values = copy.deepcopy(self._cached_values)

        # static file-based layers that already have values are not reloaded
        elif con1 and not self.model_space and self._values is not None:

            pass

        elif con1:

            with rio.open(str(self.source)) as src:

                self._values = src.read(1).squeeze()
                self.profile = src.profile

                if self.scale:

                    self._values = self._values.astype(type(self.scale))
                    data_mask = self._get_data_mask()
                    self._values[data_mask] /= self.scale

                if to_cache and self.cache:
                    self._cached_values = copy.deepcopy(self._values)

    @_doc_function_fields
    @validate_all_types
    def update(self,
        input_layers: Annotated[List[np.ndarray], Field(
            description="""List of 2D-arrays that are to be passed as input layer to the layer
                function to get updated layer values. At least one input layer must be given. All
                input layer arrays must have identical dimenions."""
        )],
        rows: Annotated[Optional[slice], Field(
            description="""Slice object that denotes the rows of the complete layer array that
                are being updated. The number of rows covered by this slice must match the number
                of rows of the passed input layers."""
        )] = slice(None),
        columns: Annotated[Optional[slice], Field(
            description="""Slice object that denotes the columns of the complete layer array that
                are being updated. The number of columns covered by this slice must match the number
                of columns of the passed input layers."""
        )] = slice(None),
        buffer: Annotated[Optional[int], Field(
            ge=0,
            description="""Buffer distance, expressed in cells, over which neighboring cells are
                included when updating layer values. Useful for context-sensitive operators.
                Only has effect if both rows and columns are specified. If specified, the number of
                rows of all passed input layers must equal len(rows slice) + 2 * buffer, and
                likewise, the number of columns must equal len(columns slice) + 2 * buffer. Will
                be forced to 0 if either rows or columns is left unspecified. Must be greater than
                or equal to 0."""
        )] = 0,
        from_cache: Annotated[Optional[bool], Field(
            description="""Whether to update values with the values stored in cache. Only has effect
                if cache=True for the layer and if there are cached values."""
        )] = False,
        to_cache: Annotated[Optional[bool], Field(
            description="""Whether to store a copy of the updated values in cache. Only has effect
                if cache=True for the layer."""
        )] = False
    ) -> None:

        """
        _summary_
        """

        input_layers = list(input_layers)

        if len(input_layers) > 1:
            for i in range(len(input_layers) - 1):
                msg = """All passed input layers must have the same shape"""
                assert input_layers[i].shape == input_layers[i + 1].shape, msg

        if rows == slice(None):
            buffer = 0
        if columns == slice(None):
            buffer = 0

        con1 = isinstance(self.source, LayerFunction)

        if con1 and from_cache and self.cache and self._cached_values is not None:
            self._values = copy.deepcopy(self._cached_values)

        elif con1:

            inputs = [l[rows, columns] for l in input_layers]
            values = self.source._evaluate(inputs).squeeze()

            if buffer > 0:
                rows = slice(rows.start + buffer, rows.stop - buffer)
                columns = slice(columns.start + buffer, columns.stop - buffer)
                values = values[buffer:-buffer, buffer:-buffer]

            if self._values is not None:
                self._values[rows, columns] = values
            else:
                self._values = values

            if to_cache and self.cache:
                self._cached_values = copy.deepcopy(self._values)

    @_doc_function_fields
    @validate_all_types  
    def get(self) -> np.ndarray:

        """
        _summary_
        """

        return self._values

    @_doc_function_fields
    @validate_all_types
    def set(self,
        values: Annotated[np.ndarray, Field(
            description="""2D-array with which the layer values are to be set. Note that if the
                dimensions of array deviate from layer profile or model space, this may cause errors
                in other operations."""
        )]
    ) -> None:

        """
        _summary_
        """
        if len(values.squeeze().shape) != 2:
            raise ValueError('Array must be two-dimensional')

        self._values = values.squeeze()

    def empty(self) -> None:

        """
        _summary_
        """

        self._values = None

    @_doc_function_fields
    def set_profile(self,
        profile: Annotated[dict, Field(
            description="""Dictionary containing the parameters needed by rasterio to write layers
                to GIS-supported file format"""
        )]
    ) -> None:

        """
        _summary_
        """

        self.profile = profile

    @_doc_function_fields
    def save(self,
        path: Annotated[Union[str, Path], Field(
            description="""File path pointing to the location in the local file system where the
                layer is to be saved"""
        )]
    ) -> None:

        """
        _summary_
        """

        if self._values is None:
            raise ValueError('Layer must have loaded or updated values to allow saving to file')

        if len(self.profile) == 0:
            raise ValueError('Layer must have a specified profile to allow saving to file')

        values = copy.deepcopy(self._values)

        if self.scale:
            data_mask = self._get_data_mask()
            values[data_mask] = values[data_mask] * self.scale

        with rio.open(str(path), 'w', **self.profile) as dst:
            dst.write(values, 1)
