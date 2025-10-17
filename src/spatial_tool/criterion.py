# spatial_tool/criterion.py
"""Module containing the ObjectiveCriterion and DecisionCriterion class definitions"""


from typing import Optional, Annotated
from pydantic import BaseModel, Field

from .function import ObjectiveFunction, DecisionFunction
from .support import _doc_class_fields


class _BaseCriterion(BaseModel):

    criterion_id: str = Field(
        description="""Unique identifier for the criterion"""
    )


@_doc_class_fields
class ObjectiveCriterion(_BaseCriterion):

    """
    _summary_
    """

    objective_function: Annotated[ObjectiveFunction, Field(
        description="""Objective function associated with the criterion"""
    )]
    maximize: Annotated[Optional[bool], Field(
        description="Whether to maximize the criterion"
    )] = False


@_doc_class_fields
class DecisionCriterion(_BaseCriterion):

    """
    _summary_
    """

    decision_function: Annotated[DecisionFunction, Field(
        description="Decision function associated with the criterion"
    )]
    cover_type: Annotated[str, Field(
        description="""Label of the cover type to which the decision function is applied"""
    )]
