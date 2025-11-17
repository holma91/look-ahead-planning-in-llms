import json
import random
from dataclasses import dataclass
from collections import deque
from typing import List, Optional, Tuple, Dict, Any


@dataclass(frozen=True)
class Action:
    """Blocksworld action: either 'pick-up X' or 'stack X on-top-of Y/table'."""

    kind: str  # "pick-up" or "stack"
    block: str  # the moved block
    target: str = ""  # another block or "table" (only used for stack)


class State:
    """
    Blocksworld state: list of stacks + what (if anything) is held in hand.
    stacks: list of lists, each list bottom->top of block colors
    holding: None or a block color
    """

    def __init__(self, stacks: List[List[str]], holding: Optional[str] = None):
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
            # can pick up any clear (top) block
            for top in self.top_blocks():
                actions.append(Action(kind="pick-up", block=top))
        else:
            # holding a block: can stack it on any clear top or on the table
            b = self.holding
            actions.append(Action(kind="stack", block=b, target="table"))
            for top in self.top_blocks():
                # can't stack on itself
                if top != b:
                    actions.append(Action(kind="stack", block=b, target=top))

        return actions

    def apply(self, action: Action) -> "State":
        """Return the next state after applying an action (assumed legal)."""
        s = self.clone()

        if action.kind == "pick-up":
            # must not be holding anything
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
                # new stack with just this block
                s.stacks.append([b])
            else:
                # place on top of target
                idx = s.find_stack_index(action.target)
                assert idx is not None
                assert s.stacks[idx][-1] == action.target  # target must be clear
                s.stacks[idx].append(b)
            s.holding = None

        return s


def random_state(colors: List[str], max_piles: int, rng: random.Random) -> State:
    """Generate a random arrangement of blocks on the table (no holding)."""
    num_piles = rng.randint(1, max_piles)
    piles = [[] for _ in range(num_piles)]
    for c in colors:
        idx = rng.randrange(num_piles)
        piles[idx].append(c)
    # remove empty piles
    piles = [p for p in piles if p]
    # randomize order within each pile
    for p in piles:
        rng.shuffle(p)
    return State(piles, holding=None)


def bfs_plan(init: State, goal: State, max_steps: int) -> Optional[List[Action]]:
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


def generate_dataset(
    num_examples: int,
    desired_steps: int,
    min_colors: int,
    max_colors: int,
    max_piles: int,
    max_steps: int,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Algorithm:
    1. Sample min_colors to max_colors colors
    2. Sample init and goal until we find a solvable pair
    3. Compute the optimal plan
    4. If plan has the same number of steps as desired_steps, add example to dataset
    """
    rng = random.Random(seed)
    dataset: List[Dict[str, Any]] = []

    while len(dataset) < num_examples:
        # 1. sample number of colors and their names
        num_colors = rng.randint(min_colors, max_colors)
        base_colors = ["red", "blue", "green", "yellow", "white", "black", "gray"]
        colors = base_colors[:num_colors]

        # 2. sample init and goal until we find a solvable pair
        init = random_state(colors, max_piles, rng)
        goal = random_state(colors, max_piles, rng)
        if init == goal:
            continue

        # 3. compute the optimal plan (up to max_steps)
        plan = bfs_plan(init, goal, max_steps=max_steps)
        if plan is None:
            continue

        # 4. if plan has less steps than desired_steps, add example to dataset
        if len(plan) == desired_steps:
            example = {
                "colors": colors,
                "init_stacks": init.stacks,
                "goal_stacks": goal.stacks,
                "plan": plan,
            }
            dataset.append(example)

    return dataset


### FORMATTING ###


def stacks_to_str(stacks: List[List[str]]) -> str:
    """
    Represent stacks like (top first):
    <green on yellow on blue>
    <red>
    <white on black>
    """
    lines = []
    for s in stacks:
        if len(s) == 1:
            lines.append(f"<{s[0]}>")
        else:
            # internally: bottom -> top
            # for display: top -> bottom (more natural in english and how it's done in the paper)
            top_first = list(reversed(s))
            lines.append("<" + " on ".join(top_first) + ">")
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

    plan_lines = [action_to_str(i + 1, act) for i, act in enumerate(example["plan"])]
    plan_str = "Plan:\n" + "\n".join(plan_lines) + "\n"

    return rule + "\n" + init_str + "\n" + goal_str + "\n" + plan_str


def save_as_text(examples: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            f.write(f"### Example {i+1}\n")
            f.write(example_to_plain_text(ex))
            f.write("\n\n")


def save_as_jsonl(examples: List[Dict[str, Any]], path: str) -> None:
    """Save dataset in JSONL format for fine-tuning."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            rule = (
                "You can pick-up color1. "
                "You can stack color1 on-top-of color2. "
                "You can stack color1 on-top-of table."
            )
            init_str = "Init state:\n" + stacks_to_str(ex["init_stacks"])
            goal_str = "Goal state:\n" + stacks_to_str(ex["goal_stacks"])

            input_text = f"Rule:\n{rule}\n\n{init_str}\n\n{goal_str}"

            plan_lines = [action_to_str(i + 1, act) for i, act in enumerate(ex["plan"])]
            output_text = "Plan:\n" + "\n".join(plan_lines)

            jsonl_entry = {
                "instruction": "Given the rules, initial state, and goal state, generate an optimal plan to transform the initial state into the goal state.",
                "input": input_text,
                "output": output_text,
            }

            f.write(json.dumps(jsonl_entry) + "\n")


if __name__ == "__main__":
    # Hyperparams from the paper (Table 1)
    min_colors = 4
    max_colors = 6
    max_piles = 4
    max_steps = 6

    # Match paper's dataset sizes (train:test = 1:3)
    # L1: 7 train + 115 test = 122 total
    # L2: 88 train + 407 test = 495 total
    # L3: 472 train + 1057 test = 1529 total
    n_L1_total = 122  # 2-step plans (easy)
    n_L2_total = 495  # 4-step plans (medium)
    n_L3_total = 1529  # 6-step plans (hard)

    print("Generating datasets to match paper (Table 1)...")

    # Generate full datasets
    print(f"Generating L1 ({n_L1_total} examples)...")
    ds_L1 = generate_dataset(
        num_examples=n_L1_total,
        desired_steps=2,
        min_colors=min_colors,
        max_colors=max_colors,
        max_piles=max_piles,
        max_steps=max_steps,
        seed=1,
    )

    print(f"Generating L2 ({n_L2_total} examples)...")
    ds_L2 = generate_dataset(
        num_examples=n_L2_total,
        desired_steps=4,
        min_colors=min_colors,
        max_colors=max_colors,
        max_piles=max_piles,
        max_steps=max_steps,
        seed=2,
    )

    print(f"Generating L3 ({n_L3_total} examples)...")
    ds_L3 = generate_dataset(
        num_examples=n_L3_total,
        desired_steps=6,
        min_colors=min_colors,
        max_colors=max_colors,
        max_piles=max_piles,
        max_steps=max_steps,
        seed=3,
    )

    # Split train/test (1:3 ratio from paper)
    # L1: first 7 train, rest test
    # L2: first 88 train, rest test
    # L3: first 472 train, rest test
    ds_L1_train, ds_L1_test = ds_L1[:7], ds_L1[7:]
    ds_L2_train, ds_L2_test = ds_L2[:88], ds_L2[88:]
    ds_L3_train, ds_L3_test = ds_L3[:472], ds_L3[472:]

    # Combine all training data (as done in paper)
    ds_train_all = ds_L1_train + ds_L2_train + ds_L3_train
    ds_test_all = ds_L1_test + ds_L2_test + ds_L3_test

    print(
        f"\nTrain: L1={len(ds_L1_train)}, L2={len(ds_L2_train)}, L3={len(ds_L3_train)}, Total={len(ds_train_all)}"
    )
    print(
        f"Test:  L1={len(ds_L1_test)}, L2={len(ds_L2_test)}, L3={len(ds_L3_test)}, Total={len(ds_test_all)}"
    )

    print("\nSaving datasets...")

    # Save per-level train/test splits (.jsonl and .txt)
    save_as_jsonl(ds_L1_train, "./data/blocksworld_L1_train.jsonl")
    save_as_jsonl(ds_L1_test, "./data/blocksworld_L1_test.jsonl")
    save_as_text(ds_L1_train, "./data/blocksworld_L1_train.txt")
    save_as_text(ds_L1_test, "./data/blocksworld_L1_test.txt")

    save_as_jsonl(ds_L2_train, "./data/blocksworld_L2_train.jsonl")
    save_as_jsonl(ds_L2_test, "./data/blocksworld_L2_test.jsonl")
    save_as_text(ds_L2_train, "./data/blocksworld_L2_train.txt")
    save_as_text(ds_L2_test, "./data/blocksworld_L2_test.txt")

    save_as_jsonl(ds_L3_train, "./data/blocksworld_L3_train.jsonl")
    save_as_jsonl(ds_L3_test, "./data/blocksworld_L3_test.jsonl")
    save_as_text(ds_L3_train, "./data/blocksworld_L3_train.txt")
    save_as_text(ds_L3_test, "./data/blocksworld_L3_test.txt")

    # Save combined train/test splits (.jsonl and .txt)
    save_as_jsonl(ds_train_all, "./data/blocksworld_train.jsonl")
    save_as_jsonl(ds_test_all, "./data/blocksworld_test.jsonl")
    save_as_text(ds_train_all, "./data/blocksworld_train.txt")
    save_as_text(ds_test_all, "./data/blocksworld_test.txt")

    print("\n" + "=" * 60)
    print("SAVED DATASETS (matching paper Table 1)")
    print("=" * 60)
    print("\nCombined (for fine-tuning):")
    print(f"  blocksworld_train.jsonl     ({len(ds_train_all)} examples)")
    print(f"  blocksworld_test.jsonl      ({len(ds_test_all)} examples)")
    print(f"  blocksworld_train.txt       ({len(ds_train_all)} examples)")
    print(f"  blocksworld_test.txt        ({len(ds_test_all)} examples)")
    print("\nPer-level splits:")
    print(f"  blocksworld_L1_train.jsonl  ({len(ds_L1_train)} examples)")
    print(f"  blocksworld_L1_test.jsonl   ({len(ds_L1_test)} examples)")
    print(f"  blocksworld_L2_train.jsonl  ({len(ds_L2_train)} examples)")
    print(f"  blocksworld_L2_test.jsonl   ({len(ds_L2_test)} examples)")
    print(f"  blocksworld_L3_train.jsonl  ({len(ds_L3_train)} examples)")
    print(f"  blocksworld_L3_test.jsonl   ({len(ds_L3_test)} examples)")
    print(f"  (+ corresponding .txt files)")
    print("=" * 60)
