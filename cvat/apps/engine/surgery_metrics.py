# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Surgery QA metrics: compute per-job and aggregate quality indicators
for program leads to monitor annotation progress and quality.
"""

from django.db.models import Count, Q, F, Sum, Case, When, IntegerField

from cvat.apps.engine.models import (
    Issue,
    Job,
    JobClassification,
    JobNarration,
    JobTranscript,
    LabeledInterval,
)


def compute_job_metrics(job: Job) -> dict:
    """Compute quality metrics for a single job."""
    start_frame = job.segment.start_frame
    stop_frame = job.segment.stop_frame
    total_frames = max(stop_frame - start_frame + 1, 1)

    # Interval coverage (using interval math, not frame sets — safe for 60fps long videos)
    intervals = list(
        LabeledInterval.objects.filter(job=job)
        .values_list("frame", "end_frame", "source")
        .order_by("frame")
    )
    interval_count = len(intervals)

    auto_count = 0
    manual_count = 0
    for _, _, source in intervals:
        if source in ('auto', 'semi-auto'):
            auto_count += 1
        else:
            manual_count += 1

    # Merge overlapping intervals to compute coverage without building frame sets
    merged = []
    for s, e, _ in sorted(intervals, key=lambda x: x[0]):
        s = max(s, start_frame)
        e = min(e, stop_frame)
        if s > e:
            continue
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    covered_frames = sum(e - s + 1 for s, e in merged)
    coverage_pct = round((covered_frames / total_frames) * 100, 1)

    # Phase change density (intervals per 1000 frames)
    density = round((interval_count / total_frames) * 1000, 2) if total_frames > 0 else 0

    # Unlabeled gaps: count spaces between merged intervals (and edges)
    gap_count = 0
    if not merged:
        gap_count = 1 if total_frames > 0 else 0
    else:
        if merged[0][0] > start_frame:
            gap_count += 1
        for i in range(1, len(merged)):
            if merged[i][0] > merged[i - 1][1] + 1:
                gap_count += 1
        if merged[-1][1] < stop_frame:
            gap_count += 1

    # Issues
    issues_qs = Issue.objects.filter(job=job)
    issue_summary = {
        "total": issues_qs.count(),
        "open": issues_qs.filter(resolved=False).count(),
        "resolved": issues_qs.filter(resolved=True).count(),
        "by_type": dict(
            issues_qs.values_list("issue_type")
            .annotate(count=Count("id"))
            .values_list("issue_type", "count")
        ),
    }

    # Narrations + transcripts
    narration_count = JobNarration.objects.filter(job=job).count()
    transcript_statuses = dict(
        JobTranscript.objects.filter(narration__job=job)
        .values_list("status")
        .annotate(count=Count("id"))
        .values_list("status", "count")
    )

    # Classifications
    classifications = list(
        JobClassification.objects.filter(job=job)
        .select_related("label")
        .values_list("label__name", flat=True)
    )

    return {
        "job_id": job.id,
        "task_id": job.segment.task_id,
        "task_name": job.segment.task.name,
        "stage": job.stage,
        "state": job.state,
        "assignee": job.assignee.username if job.assignee else None,
        "total_frames": total_frames,
        "coverage": {
            "covered_frames": covered_frames,
            "coverage_pct": coverage_pct,
            "gap_count": gap_count,
        },
        "intervals": {
            "total": interval_count,
            "manual": manual_count,
            "auto": auto_count,
            "density_per_1k": density,
        },
        "issues": issue_summary,
        "narrations": narration_count,
        "transcripts": transcript_statuses,
        "classifications": classifications,
        "completeness": {
            "has_classifications": len(classifications) > 0,
            "has_narration": narration_count > 0,
            "has_intervals": interval_count > 0,
            "has_completed_transcript": transcript_statuses.get("completed", 0) > 0,
        },
    }


def compute_project_metrics(project_id: int) -> dict:
    """Aggregate metrics across all jobs in a project."""
    jobs = Job.objects.filter(
        segment__task__project_id=project_id,
    ).select_related("segment__task", "assignee")

    job_metrics = [compute_job_metrics(job) for job in jobs]

    total_jobs = len(job_metrics)
    if total_jobs == 0:
        return {"project_id": project_id, "jobs": [], "summary": {}}

    completed = sum(1 for m in job_metrics if m["state"] == "completed")
    avg_coverage = round(
        sum(m["coverage"]["coverage_pct"] for m in job_metrics) / total_jobs, 1,
    )
    total_open_issues = sum(m["issues"]["open"] for m in job_metrics)
    total_intervals = sum(m["intervals"]["total"] for m in job_metrics)
    total_auto = sum(m["intervals"]["auto"] for m in job_metrics)
    fully_complete = sum(
        1 for m in job_metrics if all(m["completeness"].values())
    )

    return {
        "project_id": project_id,
        "summary": {
            "total_jobs": total_jobs,
            "completed_jobs": completed,
            "completion_pct": round((completed / total_jobs) * 100, 1),
            "avg_coverage_pct": avg_coverage,
            "total_open_issues": total_open_issues,
            "total_intervals": total_intervals,
            "total_auto_intervals": total_auto,
            "fully_complete_jobs": fully_complete,
        },
        "jobs": job_metrics,
    }
