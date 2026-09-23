"""The instruction and label wording shared by every contestant, built from a Task."""

from triage_bench.domain.task import Task

LLM_MAX_TOKENS = 64


def label_descriptions_text(task: Task) -> str:
    return "\n".join(f"{label.name}: {label.description}" for label in task.labels)


def llm_system_prompt(task: Task) -> str:
    heading = f"{task.label_noun.capitalize()}s"
    return (
        f"{task.instructions}\n\n{heading}:\n{label_descriptions_text(task)}\n\n"
        f"Answer with exactly one {task.label_noun} label and your confidence that it is "
        "correct, as a number from 0 to 1."
    )
