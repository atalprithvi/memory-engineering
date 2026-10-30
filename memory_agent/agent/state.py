"""Mutable state for one agent run."""

from dataclasses import dataclass, field


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().replace("_", " ").split())


@dataclass
class TaskItem:
    description: str
    status: str = "pending"


@dataclass
class Observation:
    step: int
    action: str
    text: str


@dataclass
class MemoryRecord:
    user_message: str
    assistant_message: str
    created_at: str
    score: float


@dataclass
class AgentState:
    user_request: str
    max_steps: int = 5
    tasks: list[TaskItem] = field(default_factory=list)
    current_task: str | None = None
    current_step: int = 0
    actions: list[str] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    memories: list[MemoryRecord] = field(default_factory=list)
    finished: bool = False
    final_answer: str = ""

    def add_task(self, description: str) -> None:
        self.tasks.append(TaskItem(description=description))

    def pending_tasks(self) -> list[str]:
        return [task.description for task in self.tasks if task.status == "pending"]

    def complete_task(self, description: str) -> bool:
        target = _normalize(description)
        for task in self.tasks:
            if task.status != "pending":
                continue
            if _normalize(task.description) == target:
                task.status = "completed"
                return True
        return False

    def all_tasks_completed(self) -> bool:
        return bool(self.tasks) and all(task.status == "completed" for task in self.tasks)

    def can_continue(self) -> bool:
        return not self.finished and self.current_step < self.max_steps

    def advance(self) -> None:
        self.current_step += 1

    def finish(self, answer: str) -> None:
        self.final_answer = answer
        self.finished = True

    def format_tasks(self) -> str:
        if not self.tasks:
            return "NONE"
        return "\n".join(f"- [{task.status}] {task.description}" for task in self.tasks)

    def format_pending_tasks(self) -> str:
        pending = self.pending_tasks()
        if not pending:
            return "NONE"
        return "\n".join(f"- {task}" for task in pending)

    def format_memories(self) -> str:
        if not self.memories:
            return "NONE"
        lines = []
        for memory in self.memories:
            if memory.assistant_message:
                lines.append(
                    f"- User: {memory.user_message} | Assistant: {memory.assistant_message}"
                )
            else:
                lines.append(f"- {memory.user_message}")
        return "\n".join(lines)

    def format_observations(self) -> str:
        if not self.observations:
            return "NONE"
        return "\n".join(
            f"- step {item.step}: {item.action} -> {item.text}"
            for item in self.observations
        )

    def format_list(self, items: list[str]) -> str:
        if not items:
            return "NONE"
        return "\n".join(f"- {item}" for item in items)
