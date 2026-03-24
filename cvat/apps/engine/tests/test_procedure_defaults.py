# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from django.contrib.auth import get_user_model
from django.test import TestCase

from cvat.apps.engine.models import Job, JobClassification, Label, Project, Segment, Task
from cvat.apps.engine.procedure_defaults import (
    backfill_default_job_classifications_for_task,
    ensure_default_job_classification,
)

User = get_user_model()


class ProcedureDefaultClassificationTest(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(username="owner", password="owner")

    def _create_project_job(
        self,
        *,
        project_name: str = "Cholecystectomy",
        label_names: tuple[str, ...] = ("Cholecystectomy",),
    ) -> tuple[Project, Task, dict[str, Label], Job]:
        project = Project.objects.create(name=project_name, owner=self.owner)
        labels = {
            name: Label.objects.create(project=project, name=name)
            for name in label_names
        }
        task = Task.objects.create(
            name="case-1",
            mode="interpolation",
            owner=self.owner,
            project=project,
        )
        segment = Segment.objects.create(task=task, start_frame=0, stop_frame=10)
        job = Job.objects.create(segment=segment)
        return project, task, labels, job

    def test_job_creation_auto_assigns_project_procedure_classification(self):
        _project, _task, labels, job = self._create_project_job()

        classification = JobClassification.objects.get(job=job)

        self.assertEqual(classification.label, labels["Cholecystectomy"])
        self.assertEqual(classification.owner, self.owner)

    def test_job_creation_supports_project_suffix_in_name(self):
        _project, _task, labels, job = self._create_project_job(
            project_name="Cholecystectomy project",
            label_names=("Cholecystectomy",),
        )

        classification = JobClassification.objects.get(job=job)

        self.assertEqual(classification.label, labels["Cholecystectomy"])

    def test_existing_classification_is_respected(self):
        _project, _task, labels, job = self._create_project_job(
            label_names=("Cholecystectomy", "Appendectomy"),
        )
        JobClassification.objects.filter(job=job).delete()
        JobClassification.objects.create(
            job=job,
            label=labels["Appendectomy"],
            owner=self.owner,
        )

        result = ensure_default_job_classification(job)

        self.assertIsNone(result)
        self.assertFalse(
            JobClassification.objects.filter(job=job, label=labels["Cholecystectomy"]).exists()
        )
        self.assertTrue(
            JobClassification.objects.filter(job=job, label=labels["Appendectomy"]).exists()
        )

    def test_backfill_applies_default_after_task_is_added_to_project(self):
        project = Project.objects.create(name="Cholecystectomy", owner=self.owner)
        label = Label.objects.create(project=project, name="Cholecystectomy")
        task = Task.objects.create(name="case-2", mode="interpolation", owner=self.owner)
        segment = Segment.objects.create(task=task, start_frame=0, stop_frame=10)
        job = Job.objects.create(segment=segment)

        self.assertFalse(JobClassification.objects.filter(job=job).exists())

        task.project = project
        task.save(update_fields=["project", "updated_date"])

        created = backfill_default_job_classifications_for_task(task)

        self.assertEqual(created, 1)
        self.assertTrue(JobClassification.objects.filter(job=job, label=label).exists())
