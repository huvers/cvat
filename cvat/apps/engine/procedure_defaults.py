import re

from cvat.apps.engine.models import Job, JobClassification, Label, Project, Task

_PROJECT_SUFFIX_RE = re.compile(r"\s+project$", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_name(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value.strip())


def _candidate_procedure_names(task: Task) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    def add(name: str | None) -> None:
        if not name:
            return

        cleaned = _clean_name(name)
        if not cleaned:
            return

        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            names.append(cleaned)

    dataset_episode = task.dataset_episodes.select_related("dataset").first()
    if dataset_episode is not None:
        add(dataset_episode.dataset.procedure_type)

    if task.project_id:
        normalized_project_name = _PROJECT_SUFFIX_RE.sub("", task.project.name).strip()
        add(normalized_project_name)
        add(task.project.name)

    return names


def resolve_default_procedure_label(task: Task) -> Label | None:
    for candidate in _candidate_procedure_names(task):
        if task.project_id:
            label = Label.objects.filter(
                project_id=task.project_id,
                parent__isnull=True,
                name__iexact=candidate,
            ).first()
            if label is not None:
                return label

        label = Label.objects.filter(
            task_id=task.id,
            parent__isnull=True,
            name__iexact=candidate,
        ).first()
        if label is not None:
            return label

    return None


def ensure_default_job_classification(job: Job) -> JobClassification | None:
    if JobClassification.objects.filter(job_id=job.id).exists():
        return None

    task = job.segment.task
    label = resolve_default_procedure_label(task)
    if label is None:
        return None

    owner = task.owner
    if owner is None and task.project_id:
        owner = task.project.owner

    classification, _ = JobClassification.objects.get_or_create(
        job=job,
        label=label,
        defaults={"owner": owner},
    )
    return classification


def backfill_default_job_classifications_for_task(task: Task) -> int:
    created = 0
    for job in Job.objects.filter(segment__task_id=task.id).select_related("segment__task__project"):
        if ensure_default_job_classification(job) is not None:
            created += 1

    return created


def backfill_default_job_classifications_for_project(project: Project) -> int:
    created = 0
    for task in Task.objects.filter(project_id=project.id).select_related("project", "owner"):
        created += backfill_default_job_classifications_for_task(task)

    return created
