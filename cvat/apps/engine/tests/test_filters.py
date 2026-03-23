# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from unittest import mock

from django.test import SimpleTestCase

from cvat.apps.engine.filters import JsonLogicFilter


class JsonLogicFilterTest(SimpleTestCase):
    def test_apply_filter_supports_not_equal(self):
        queryset = mock.Mock()
        queryset.filter.return_value = queryset

        result = JsonLogicFilter().apply_filter(
            queryset,
            {"!=": [{"var": "state"}, "completed"]},
            lookup_fields={"state": "state"},
        )

        self.assertIs(result, queryset)
        queryset.filter.assert_called_once()

        q_object = queryset.filter.call_args.args[0]
        self.assertTrue(q_object.negated)
        self.assertEqual([("state", "completed")], q_object.children)
