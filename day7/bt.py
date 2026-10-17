"""
A small behavior-tree library (the class's "automated decision making"
algorithm we chose). A tree is ticked many times a second; each tick walks
the tree top-down, left-to-right, and the first branch whose condition holds
gets to act.

    Selector  -- tries children in order, returns the first that is not FAILURE
                 (this is where priorities live: edge safety before attacking
                 before obeying the driver before searching)
    Sequence  -- runs children in order, stops at the first FAILURE
    Condition -- a yes/no question about the blackboard (sensor fusion lives here)
    Action    -- does something; may stay RUNNING across ticks (a timed retreat)

The blackboard is just a dict the sensors write into and the nodes read.
"""

SUCCESS, FAILURE, RUNNING = "SUCCESS", "FAILURE", "RUNNING"


class Node:
    name = "node"

    def tick(self, bb):
        raise NotImplementedError


class Condition(Node):
    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def tick(self, bb):
        return SUCCESS if self.fn(bb) else FAILURE


class Action(Node):
    """fn(bb) returns SUCCESS / FAILURE / RUNNING."""

    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def tick(self, bb):
        bb["active"] = self.name
        return self.fn(bb)


class Sequence(Node):
    def __init__(self, name, *children):
        self.name, self.children = name, list(children)

    def tick(self, bb):
        for c in self.children:
            r = c.tick(bb)
            if r != SUCCESS:
                return r
        return SUCCESS


class Selector(Node):
    def __init__(self, name, *children):
        self.name, self.children = name, list(children)

    def tick(self, bb):
        for c in self.children:
            r = c.tick(bb)
            if r != FAILURE:
                bb.setdefault("trace", []).append(c.name)
                return r
        return FAILURE


def describe(node, depth=0):
    """Text rendering of the tree, for the Notion page and the HUD."""
    kind = type(node).__name__
    lines = ["  " * depth + f"{kind}: {node.name}"]
    for c in getattr(node, "children", []):
        lines += describe(c, depth + 1)
    return lines
