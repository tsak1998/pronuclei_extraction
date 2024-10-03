from enum import StrEnum, Enum
from typing import Literal, Optional
from pydantic import BaseModel
from scipy.stats import rv_continuous, rv_discrete


class GrowthEventsEnum(StrEnum):
    tPNf = "tPNf"
    t2 = "t2"
    t3 = "t3"
    t4 = "t4"
    t5 = "t5"
    t6 = "t6"
    t7 = "t7"
    t8 = "t8"
    t9 = "t9"
    tM = "tM"


class Material(BaseModel):
    E: float


class Position(BaseModel):
    x: float
    y: float
    z: float


class Cell(BaseModel):
    cell_id: int
    position: Position


class EmbryoGrowthEvent(BaseModel):
    event: Literal[GrowthEventsEnum]
    time: float


class Embryo(BaseModel):
    cells: list[Cell]
    division_times: list[EmbryoGrowthEvent]


class DistributionParams(BaseModel):
    shape: Optional[float]
    loc: Optional[float]
    scale: Optional[float]


class Distribution(BaseModel):
    """
    Attributes
    distribution (str): the theoretical distribution
    params (DistributionParams): the parameters of the fitted distribution
    params (rv_continuous | rv_discrete): the fitted scipy distribution
    """
    distribution: str
    params: DistributionParams
    fitted_dist: rv_continuous | rv_discrete
