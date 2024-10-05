from pydantic import BaseModel


class Config(BaseModel):
    simul_frames: int
    fps: int
    experiment_path: str
