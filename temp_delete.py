from dataclasses import dataclass, field


@dataclass
class Metric:
    values: list[float] = field(default_factory=list)
    running_total: float = 0.0
    num_updates: float = 0.0
    average: float = 0.0


obj1 = Metric()
obj1.num_updates = 1

obj2 = Metric()

print(obj1)
print(obj2)
