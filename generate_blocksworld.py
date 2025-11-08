import json
import random
from dataclasses import dataclass
from collections import deque
from typing import List, Optional, Tuple, Dict, Any


# -----------------------------
#  Blocksworld core structures
# -----------------------------

@dataclass(frozen=True)
class Action:
    """Blocksworld action: either 'pick-up X' or 'stack X on-top-of Y/table'."""
    kind: str           # "pick-up" or "stack"
    block: str          # the moved block
    target: str = ""    # another block or "table" (only used for stack)


class State:
    """
    Blocksworld state: list of stacks + what (if anything) is held in hand.
    stacks: list of lists, each list bottom->top of block colors
    holding: None or a block color
    """
    def __init__(self, stacks: List[List[str]], holding: Optional[str] = None):
        # Deep copy to avoid aliasing
        self.stacks = [list(s) for s in stacks]
        self.holding = holding

    def clone(self) -> "State":
        return State(self.stacks, self.holding)

    def __repr__(self) -> str:
        return f"State(stacks={self.stacks}, holding={self.holding})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return False
        return self.stacks == other.stacks and self.holding == other.holding

    def __hash__(self) -> int:
        stacks_tuple = tuple(tuple(s) for s in self.stacks)
        return hash((stacks_tuple, self.holding))

    # --------- legality & transitions ----------

    def top_blocks(self) -> List[str]:
        """Return a list of top blocks of each non-empty stack."""
        return [s[-1] for s in self.stacks if s]

    def find_stack_index(self, block: str) -> Optional[int]:
        """Return index of stack containing block, or None."""
        for i, s in enumerate(self.stacks):
            if block in s:
                return i
        return None

    def legal_actions(self) -> List[Action]:
        """All legal actions from this state."""
        actions: List[Action] = []

        if self.holding is None:
            # Can pick up any clear (top) block
            for top in self.top_blocks():
                actions.append(Action(kind="pick-up", block=top))
        else:
            # Holding a block: can stack it on any clear top or on the table
            b = self.holding
            actions.append(Action(kind="stack", block=b, target="table"))
            for top in self.top_blocks():
                # Can't stack on itself
                if top != b:
                    actions.append(Action(kind="stack", block=b, target=top))

        return actions

    def apply(self, action: Action) -> "State":
        """Return the next state after applying an action (assumed legal)."""
        s = self.clone()

        if action.kind == "pick-up":
            # Must not be holding anything
            assert s.holding is None
            idx = s.find_stack_index(action.block)
            assert idx is not None
            stack = s.stacks[idx]
            assert stack[-1] == action.block  # must be clear
            stack.pop()
            if not stack:
                s.stacks.pop(idx)
            s.holding = action.block

        elif action.kind == "stack":
            assert s.holding == action.block
            b = action.block
            if action.target == "table":
                # New stack with just this block
                s.stacks.append([b])
            else:
                # Place on top of target
                idx = s.find_stack_index(action.target)
                assert idx is not None
                assert s.stacks[idx][-1] == action.target  # target must be clear
                s.stacks[idx].append(b)
            s.holding = None

        return s


# -----------------------------
#  Random state & plan search
# -----------------------------

def random_state(colors: List[str],
                 max_piles: int,
                 rng: random.Random) -> State:
    """Generate a random arrangement of blocks on the table (no holding)."""
    num_piles = rng.randint(1, max_piles)
    piles = [[] for _ in range(num_piles)]
    for c in colors:
        idx = rng.randrange(num_piles)
        piles[idx].append(c)
    # Remove empty piles
    piles = [p for p in piles if p]
    # Randomize order within each pile
    for p in piles:
        rng.shuffle(p)
    return State(piles, holding=None)


def bfs_plan(init: State,
             goal: State,
             max_steps: int) -> Optional[List[Action]]:
    """BFS for shortest plan from init to goal, up to max_steps."""
    queue = deque([(init, [])])
    visited = {init}

    while queue:
        state, path = queue.popleft()

        if state == goal and state.holding is None:
            return path

        if len(path) >= max_steps:
            continue

        for act in state.legal_actions():
            nxt = state.apply(act)
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [act]))

    return None


# -----------------------------
#  Serialization
# -----------------------------

def stacks_to_str(stacks: List[List[str]]) -> str:
    """
    Represent stacks like:
    <white on gray>
    <red>
    <blue on green>
    """
    lines = []
    for s in stacks:
        if len(s) == 1:
            lines.append(f"<{s[0]}>")
        else:
            # bottom to top
            lines.append("<" + " on ".join(s) + ">")
    return "\n".join(lines)


def action_to_str(step_idx: int, act: Action) -> str:
    if act.kind == "pick-up":
        return f"step {step_idx}: pick-up {act.block}"
    else:
        return f"step {step_idx}: stack {act.block} on-top-of {act.target}"


def example_to_plain_text(example: Dict[str, Any]) -> str:
    """Render a dataset example into a text prompt-style format."""
    rule = (
        "Rule:\n"
        "You can pick-up color1. "
        "You can stack color1 on-top-of color2. "
        "You can stack color1 on-top-of table.\n"
    )

    init_str = "Init state:\n" + stacks_to_str(example["init_stacks"]) + "\n"
    goal_str = "Goal state:\n" + stacks_to_str(example["goal_stacks"]) + "\n"

    plan_lines = [
        action_to_str(i + 1, act) for i, act in enumerate(example["plan"])
    ]
    plan_str = "Plan:\n" + "\n".join(plan_lines) + "\n"

    return rule + "\n" + init_str + "\n" + goal_str + "\n" + plan_str


def example_to_json(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert internal example format into a JSON-serializable dict."""
    return {
        "colors": example["colors"],
        "init_stacks": example["init_stacks"],
        "goal_stacks": example["goal_stacks"],
        "plan": [
            {
                "step": i + 1,
                "kind": act.kind,
                "block": act.block,
                "target": act.target if act.kind == "stack" else None,
            }
            for i, act in enumerate(example["plan"])
        ],
    }


# -----------------------------
#  Dataset generation
# -----------------------------

def generate_dataset(num_examples: int,
                     min_colors: int = 4,
                     max_colors: int = 6,
                     max_piles: int = 4,
                     max_steps: int = 6,
                     seed: int = 0) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    dataset: List[Dict[str, Any]] = []

    for _ in range(num_examples):
        # sample number of colors and their names
        num_colors = rng.randint(min_colors, max_colors)
        base_colors = ["red", "blue", "green", "yellow", "white", "black", "gray"]
        colors = base_colors[:num_colors]

        # sample init and goal until we find a solvable pair
        while True:
            init = random_state(colors, max_piles, rng)
            goal = random_state(colors, max_piles, rng)
            if init == goal:
                continue
            plan = bfs_plan(init, goal, max_steps=max_steps)
            if plan and 1 <= len(plan) <= max_steps:
                example = {
                    "colors": colors,
                    "init_stacks": init.stacks,
                    "goal_stacks": goal.stacks,
                    "plan": plan,
                }
                dataset.append(example)
                break

    return dataset


# -----------------------------
#  Save helpers
# -----------------------------

def save_as_text(examples: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            f.write(f"### Example {i+1}\n")
            f.write(example_to_plain_text(ex))
            f.write("\n\n")


def save_as_json(examples: List[Dict[str, Any]], path: str) -> None:
    json_list = [example_to_json(ex) for ex in examples]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_list, f, indent=2)


# -----------------------------
#  Main (usage example)
# -----------------------------

if __name__ == "__main__":
    # Example usage: generate 100 examples and save as both text and JSON.
    num_examples = 100
    max_steps = 6

    ds = generate_dataset(
        num_examples=num_examples,
        min_colors=4,
        max_colors=6,
        max_piles=4,
        max_steps=max_steps,
        seed=42,
    )

    save_as_text(ds, "blocksworld_dataset.txt")
    save_as_json(ds, "blocksworld_dataset.json")
    print
