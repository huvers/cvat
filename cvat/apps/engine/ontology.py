# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Ontology versioning: immutable snapshots of a project's label schema.

Each time a project's labels change, a new OntologyVersion is created
containing the full serialized label tree. Versions can be compared
to see what changed (added/removed/renamed labels and attributes).
"""

import logging

from django.db import transaction
from django.db.models import Max

from cvat.apps.engine.models import (
    AttributeSpec,
    Label,
    OntologyVersion,
    Project,
    User,
)

logger = logging.getLogger(__name__)


def _serialize_attribute(attr: AttributeSpec) -> dict:
    return {
        "id": attr.id,
        "name": attr.name,
        "mutable": attr.mutable,
        "input_type": attr.input_type,
        "default_value": attr.default_value,
        "values": attr.values,
    }


def _serialize_label(label: Label) -> dict:
    return {
        "id": label.id,
        "name": label.name,
        "color": label.color,
        "type": label.type,
        "parent_id": label.parent_id,
        "attributes": [
            _serialize_attribute(attr)
            for attr in label.attributespec_set.all()
        ],
        "sublabels": [
            _serialize_label(sub)
            for sub in label.sublabels.all()
        ],
    }


def snapshot_project_schema(project: Project) -> dict:
    """Serialize the current label schema of a project."""
    labels = (
        Label.objects.filter(project=project, parent__isnull=True)
        .prefetch_related("attributespec_set", "sublabels__attributespec_set")
        .order_by("id")
    )
    return {
        "labels": [_serialize_label(label) for label in labels],
    }


@transaction.atomic
def create_ontology_version(
    project: Project,
    *,
    description: str = "",
    created_by: User | None = None,
) -> OntologyVersion:
    """Create a new immutable version snapshot of the project's labels."""
    schema = snapshot_project_schema(project)

    # Get next version number
    max_version = (
        OntologyVersion.objects.filter(project=project)
        .aggregate(max_v=Max("version"))["max_v"]
    ) or 0
    next_version = max_version + 1

    version = OntologyVersion.objects.create(
        project=project,
        version=next_version,
        schema=schema,
        description=description,
        created_by=created_by,
    )

    logger.info(
        "Created ontology version v%d for project %d (%d labels)",
        next_version, project.id, len(schema["labels"]),
    )
    return version


def compute_diff(old_schema: dict, new_schema: dict) -> dict:
    """
    Compare two ontology schemas and return a structured diff.

    Returns::

        {
            "added": [{"name": "NewPhase", ...}],
            "removed": [{"name": "OldPhase", ...}],
            "renamed": [{"old_name": "Phase1", "new_name": "Preparation", "id": 5}],
            "attributes_changed": [
                {"label": "Phase1", "added": [...], "removed": [...]}
            ],
        }
    """
    old_labels = {l["id"]: l for l in old_schema.get("labels", [])}
    new_labels = {l["id"]: l for l in new_schema.get("labels", [])}

    old_ids = set(old_labels.keys())
    new_ids = set(new_labels.keys())

    added = [new_labels[lid] for lid in (new_ids - old_ids)]
    removed = [old_labels[lid] for lid in (old_ids - new_ids)]

    renamed = []
    attributes_changed = []

    for lid in old_ids & new_ids:
        old = old_labels[lid]
        new = new_labels[lid]

        if old["name"] != new["name"]:
            renamed.append({
                "id": lid,
                "old_name": old["name"],
                "new_name": new["name"],
            })

        old_attrs = {a["name"]: a for a in old.get("attributes", [])}
        new_attrs = {a["name"]: a for a in new.get("attributes", [])}

        attr_added = [new_attrs[n] for n in set(new_attrs) - set(old_attrs)]
        attr_removed = [old_attrs[n] for n in set(old_attrs) - set(new_attrs)]

        if attr_added or attr_removed:
            attributes_changed.append({
                "label": new["name"],
                "label_id": lid,
                "added": attr_added,
                "removed": attr_removed,
            })

    return {
        "added": added,
        "removed": removed,
        "renamed": renamed,
        "attributes_changed": attributes_changed,
    }
