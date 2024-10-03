from typing import Optional, Union
from pydantic import BaseModel
from models import Cell


class ArrestEvent(BaseModel):
    """
    The embryo has arrested and the simulation is finished
    """


class DirectCleavage(BaseModel):
    """
    """
    direct_cleavage: bool


class Fragmentation(BaseModel):
    """
    Was there fragmentation in the cell division event
    """

    fragmentation: bool
    ratio: bool


class CellDivisionEvent(BaseModel):
    """
    When a cell division event occures.
    Until tM follow the fitted distribution.
    After that they follow the fitted growth model.
    """
    parent_cell: Cell
    child_cell: Optional[list[Cell]]
    children_mass_ratio: list[float]
    direct_cleavage: DirectCleavage
    fragmentation: Fragmentation


class SimulationEvents(BaseModel):
    event: Union[ArrestEvent, CellDivisionEvent]
    time: float
    timestep: int
