"""Game registry. Add a new game by importing its adapter and appending an instance."""

from .matching import MatchingAdapter
from .maze import MazeAdapter

REGISTRY = {a.id: a for a in (MatchingAdapter(), MazeAdapter())}
