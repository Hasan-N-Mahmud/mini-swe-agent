"""Resume a run from a recorded (and usually hand-modified) trajectory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from minisweagent.agents.default import DefaultAgent, NonTerminatingException, TerminatingException

ACTION_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)

HEREDOC_WRITE_RE = re.compile(
    r"""cat\s*(?:
            <<-?\s*['"]?[A-Za-z_]\w*['"]?\s*>>?\s*(?P<path_a>[^\s;&|]+)
          | >>?\s*(?P<path_b>[^\s;&|]+)\s*<<-?\s*['"]?[A-Za-z_]\w*['"]?
        )""",
    re.VERBOSE,
)

QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")

# Commands that change container state. The prefix must contain none of these,
MUTATING_PATTERNS = [
    (r"\bsed\s+-[a-zA-Z]*i", "sed -i"),
    (r"(?<![0-9<>&])>>?\s*[^\s|&;]", "output redirect"),
    (r"\b(rm|mv|cp|touch|mkdir|ln)\b", "file mutation"),
    (r"\bgit\s+(checkout|apply|stash|reset|add|commit|clean|revert)\b", "git mutation"),
    (r"\b(pip|conda|apt-get|npm)\s+(install|uninstall|remove)\b", "package install"),
    (r"\bpatch\b\s+-", "patch"),
    (r"\btee\b", "tee"),
]

# Not mutating on their own, but a script or test run can write files.
SUSPECT_PATTERNS = [
    (r"\bpython[0-9.]*\s+[^\s-][^\s]*\.py\b", "runs a python script"),
    (r"(^|\s)(pytest|python[0-9.]*\s+-m\s+pytest)\b", "runs pytest"),
]

SUBMIT_MARKERS = ("MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")


class ResumeError(Exception):
    """Raised when a trajectory cannot be turned into a valid resume point."""


def extract_action(message: dict) -> str | None:
    """Return the single bash action in an assistant message, or None."""
    actions = ACTION_RE.findall(message.get("content", ""))
    return actions[0].strip() if len(actions) == 1 else None


def heredoc_target(action: str) -> str | None:
    """Return the path a heredoc in `action` writes to, or None."""
    if match := HEREDOC_WRITE_RE.search(action):
        return match.group("path_a") or match.group("path_b")
    return None


def find_write_steps(messages: list[dict], suffix: str = ".py") -> list[tuple[int, str]]:
    """Every assistant step whose action creates a `suffix` file via heredoc."""
    steps = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        action = extract_action(message)
        if action and (path := heredoc_target(action)) and path.endswith(suffix):
            steps.append((index, path))
    return steps


def check_prefix_is_read_only(messages: list[dict]) -> tuple[list[str], list[str]]:
    """Classify prefix actions. Returns (mutating, suspect) descriptions."""
    mutating, suspect = [], []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        action = extract_action(message)
        if not action:
            continue
        # Strip quoted strings so that e.g. `grep "pytest-xdist"` is not a hit.
        bare = QUOTED_RE.sub("", action)
        head = action.strip().splitlines()[0][:90]
        for pattern, label in MUTATING_PATTERNS:
            if re.search(pattern, bare):
                mutating.append(f"msg {index}: {label}: {head}")
        for pattern, label in SUSPECT_PATTERNS:
            if re.search(pattern, bare):
                suspect.append(f"msg {index}: {label}: {head}")
    return mutating, suspect


def load_resume_point(
    trajectory_path: str | Path, resume_at: int | None = None, strict: bool = True
) -> tuple[list[dict], dict, dict]:
    """Split a trajectory into (prefix messages, injected action, metadata).

    `resume_at` is the index of the assistant message to inject; it defaults to
    the first step that writes a `.py` file, which in these trajectories is the
    agent's reproduction script.
    """
    trajectory_path = Path(trajectory_path)
    messages = json.loads(trajectory_path.read_text())["messages"]

    write_steps = find_write_steps(messages)
    if resume_at is None:
        if not write_steps:
            raise ResumeError(f"{trajectory_path}: no heredoc .py write found; pass an explicit resume index")
        resume_at = write_steps[0][0]

    if not 0 < resume_at < len(messages):
        raise ResumeError(f"{trajectory_path}: resume index {resume_at} out of range")
    injected = messages[resume_at]
    if injected.get("role") != "assistant":
        raise ResumeError(f"{trajectory_path}: message {resume_at} is {injected.get('role')}, expected assistant")
    action = extract_action(injected)
    if action is None:
        raise ResumeError(f"{trajectory_path}: message {resume_at} does not contain exactly one bash action")
    if messages[resume_at - 1].get("role") != "user":
        raise ResumeError(f"{trajectory_path}: message {resume_at - 1} is not an observation")

    prefix = messages[:resume_at]
    if len(prefix) < 2 or prefix[0]["role"] != "system" or prefix[1]["role"] != "user":
        raise ResumeError(f"{trajectory_path}: prefix must start with a system and an instance message")

    mutating, suspect = check_prefix_is_read_only(prefix)
    if mutating and strict:
        raise ResumeError(
            f"{trajectory_path}: prefix is not read-only, so seeding it without execution would "
            "desync the context from the container:\n  " + "\n  ".join(mutating)
        )
    for index, message in enumerate(prefix):
        if message.get("role") != "assistant":
            continue  # the instance prompt documents the submit command; only actions matter
        prefix_action = extract_action(message) or ""
        if any(marker in prefix_action for marker in SUBMIT_MARKERS):
            raise ResumeError(f"{trajectory_path}: prefix submits at msg {index}")

    metadata = {
        "trajectory": str(trajectory_path),
        "resume_at": resume_at,
        "resume_action_target": heredoc_target(action),
        "prefix_steps": sum(1 for m in prefix if m["role"] == "assistant"),
        "write_step_candidates": [{"index": i, "path": p} for i, p in write_steps],
        "prefix_mutating": mutating,
        "prefix_suspect": suspect,
    }
    return prefix, injected, metadata


class ResumedAgent(DefaultAgent):
    """DefaultAgent that starts from a seeded message history and one forced action."""

    def __init__(self, *args, prefix_messages: list[dict], injected_action: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_messages = prefix_messages
        self.injected_action = injected_action
        # Seeded steps cost no model calls, so without this the resumed run would
        # get a larger effective budget than the trajectory it continues.
        seeded = sum(1 for m in prefix_messages if m["role"] == "assistant") + 1
        if self.config.step_limit > 0:
            self.config.step_limit = max(1, self.config.step_limit - seeded)

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = [dict(message) for message in self.prefix_messages]
        pending = self.injected_action
        while True:
            try:
                if pending is None:
                    self.step()
                else:
                    self.add_message("assistant", pending["content"])
                    self.get_observation(pending)
                    pending = None
            except NonTerminatingException as e:
                pending = None
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)


def main() -> None:
    """Print the candidate resume points of a trajectory, for picking `--resume-at`."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("trajectory")
    parser.add_argument("--resume-at", type=int, default=None)
    args = parser.parse_args()

    messages = json.loads(Path(args.trajectory).read_text())["messages"]
    print(f"{len(messages)} messages\n\ncandidate resume points (heredoc .py writes):")
    for index, path in find_write_steps(messages):
        print(f"  msg {index:4d}  {path}")

    _, injected, metadata = load_resume_point(args.trajectory, args.resume_at, strict=False)
    print(f"\nselected msg {metadata['resume_at']} -> {metadata['resume_action_target']}")
    print(f"prefix: {metadata['prefix_steps']} steps")
    for label in ("prefix_mutating", "prefix_suspect"):
        for line in metadata[label]:
            print(f"  {label.split('_')[1].upper()}: {line}")
    print("\ninjected action:\n" + extract_action(injected))


if __name__ == "__main__":
    main()
