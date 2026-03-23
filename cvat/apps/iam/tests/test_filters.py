# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from django.test import SimpleTestCase

from cvat.apps.iam.filters import ORGANIZATION_OPEN_API_PARAMETERS, OrganizationFilterBackend


class OrganizationFilterBackendTest(SimpleTestCase):
    def test_schema_parameters_are_empty_for_views_without_organization_metadata(self):
        class ViewWithoutOrganizationMetadata:
            pass

        backend = OrganizationFilterBackend()

        self.assertEqual([], backend.get_schema_operation_parameters(ViewWithoutOrganizationMetadata()))

    def test_schema_parameters_are_exposed_for_collection_views_with_organization_metadata(self):
        class OrganizationCollectionView:
            iam_organization_field = "organization"
            detail = False

        backend = OrganizationFilterBackend()

        self.assertEqual(
            [parameter["name"] for parameter in backend.get_schema_operation_parameters(OrganizationCollectionView())],
            [parameter.name for parameter in ORGANIZATION_OPEN_API_PARAMETERS],
        )
