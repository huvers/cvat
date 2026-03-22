// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useEffect, useCallback } from 'react';
import { useSelector } from 'react-redux';
import Button from 'antd/lib/button';
import Select from 'antd/lib/select';
import Spin from 'antd/lib/spin';
import Tag from 'antd/lib/tag';
import notification from 'antd/lib/notification';
import { PlusOutlined } from '@ant-design/icons';

import { Job } from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import serverProxy from 'cvat-core/src/server-proxy';
import { SerializedJobClassification } from 'cvat-core/src/server-response-types';

import './styles.scss';

export default function VideoClassificationEditor(): JSX.Element {
    const job = useSelector((state: CombinedState) => state.annotation.job.instance) as Job | null | undefined;
    const labels = useSelector((state: CombinedState) => state.annotation.job.labels);

    const [classifications, setClassifications] = useState<SerializedJobClassification[]>([]);
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [selectedLabelId, setSelectedLabelId] = useState<number | undefined>(undefined);

    useEffect(() => {
        if (!job) return undefined;
        let cancelled = false;
        setLoading(true);
        serverProxy.jobs
            .getClassifications(job.id)
            .then((data) => {
                if (!cancelled) setClassifications(data);
            })
            .catch((err: unknown) => {
                if (!cancelled) notification.error({ message: 'Failed to load classifications', description: String(err) });
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });
        return () => { cancelled = true; };
    }, [job?.id]);

    const appliedLabelIds = new Set(classifications.map((c) => c.label_id));

    const availableLabels = labels.filter((l) => l.id !== undefined && !appliedLabelIds.has(l.id as number));

    const addClassification = useCallback(async () => {
        if (!job || selectedLabelId === undefined) return;
        setSaving(true);
        try {
            const created = await serverProxy.jobs.createClassification(job.id, selectedLabelId);
            setClassifications((prev) => [...prev, created]);
            setSelectedLabelId(undefined);
        } catch (err: unknown) {
            notification.error({ message: 'Failed to add classification', description: String(err) });
        } finally {
            setSaving(false);
        }
    }, [job, selectedLabelId]);

    const removeClassification = useCallback(async (labelId: number) => {
        if (!job) return;
        setSaving(true);
        try {
            await serverProxy.jobs.deleteClassification(job.id, labelId);
            setClassifications((prev) => prev.filter((c) => c.label_id !== labelId));
        } catch (err: unknown) {
            notification.error({ message: 'Failed to remove classification', description: String(err) });
        } finally {
            setSaving(false);
        }
    }, [job]);

    return (
        <div className='cvat-video-classification-editor'>
            {loading ? (
                <div className='cvat-video-classification-spinner'>
                    <Spin size='small' />
                </div>
            ) : (
                <>
                    <div className='cvat-video-classification-tags'>
                        {classifications.length === 0 ? (
                            <p className='cvat-video-classification-empty'>
                                No video-level classifications. Use the control below to tag this procedure.
                            </p>
                        ) : (
                            classifications.map((c) => (
                                <Tag
                                    key={c.id}
                                    color={c.label_color || undefined}
                                    closable={!saving}
                                    onClose={(e) => {
                                        e.preventDefault();
                                        removeClassification(c.label_id);
                                    }}
                                >
                                    {c.label_name}
                                </Tag>
                            ))
                        )}
                    </div>
                    <div className='cvat-video-classification-add'>
                        <Select
                            className='cvat-video-classification-select'
                            placeholder='Select label'
                            value={selectedLabelId}
                            onChange={(v: number) => setSelectedLabelId(v)}
                            disabled={availableLabels.length === 0 || saving}
                        >
                            {availableLabels.map((label) => (
                                <Select.Option key={label.id} value={label.id}>
                                    <span
                                        className='cvat-video-classification-swatch'
                                        style={{ backgroundColor: label.color }}
                                    />
                                    {label.name}
                                </Select.Option>
                            ))}
                        </Select>
                        <Button
                            type='primary'
                            icon={<PlusOutlined />}
                            disabled={selectedLabelId === undefined || saving}
                            onClick={addClassification}
                        >
                            Add
                        </Button>
                    </div>
                </>
            )}
        </div>
    );
}
