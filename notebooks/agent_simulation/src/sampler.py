from models import Distribution, GrowthEventsEnum

growth_event_distribution: dict[GrowthEventsEnum, Distribution] = {}


class Sampler:

    def __init__(self) -> None:
        pass

    def sample_event(self):
        raise NotImplementedError
