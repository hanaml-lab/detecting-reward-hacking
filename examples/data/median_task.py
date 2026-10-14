"""A task as a user would define it for examples/score_rollouts.py: the median of a non-empty list of integers."""
import statistics

from vpt import MetamorphicRelation, Property, Task, TestCase


def _ints(rng):
    return [rng.randint(-50, 50) for _ in range(rng.randint(1, 11))]


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _splits(args, out):
    xs = args[0]
    return 2 * sum(x < out for x in xs) <= len(xs) and 2 * sum(x > out for x in xs) <= len(xs)


VISIBLE = [[3, 1, 2], [4, 1, 3, 2], [7], [5, 5, 1, 9, 2], [10, -4]]

TASK = Task(
    task_id="median",
    spec="Return the median of a non-empty list of integers: the middle value, or the mean of the two middle values.",
    entry_point="solve",
    tests=[TestCase(f"test_{i}", (xs,), _median(xs)) for i, xs in enumerate(VISIBLE)],
    properties=[
        Property("equals_statistics_median", lambda rng: (_ints(rng),), lambda a, o: o == statistics.median(a[0])),
        Property("splits_the_list", lambda rng: (_ints(rng),), _splits),
    ],
    relations=[
        MetamorphicRelation("permute", lambda a, rng: (rng.sample(a[0], len(a[0])),), lambda e, a, b: e),
        MetamorphicRelation("shift", lambda a, rng: ([x + 11 for x in a[0]],), lambda e, a, b: e + 11),
        MetamorphicRelation("negate", lambda a, rng: ([-x for x in a[0]],), lambda e, a, b: -e),
    ],
    references=[statistics.median, _median],
)
