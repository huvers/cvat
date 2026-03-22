// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useEffect, useCallback } from 'react';
import { useSelector } from 'react-redux';
import Button from 'antd/lib/button';
import InputNumber from 'antd/lib/input-number';
import Select from 'antd/lib/select';
import Spin from 'antd/lib/spin';
import Tooltip from 'antd/lib/tooltip';
import notification from 'antd/lib/notification';
import { DeleteOutlined, PlusOutlined } from '@ant-design/icons';

import { Job } from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import { Source } from 'cvat-core/src/enums';
import serverProxy from 'cvat-core/src/server-proxy';
import { SerializedInterval } from 'cvat-core/src/server-response-types';

import './styles.scss';

export default function PhaseTrackEditor(): JSX.Element {
    const job = useSelector((state: CombinedState) => state.annotation.job.instance) as Job | null | undefined;
    const labels = useSelector((state: CombinedState) => state.annotation.job.labels);
    const currentFrame = useSelector((state: CombinedState) => state.annotation.player.frame.number);

    const [intervals, setIntervals] = useState<SerializedInterval[]>([]);
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [selectedLabelId, setSelectedLabelId] = useState<number | null>(null);
    const [startFrame, setStartFrame] = useState<number | null>(null);
    const [endFrame, setEndFrame] = useState<number | null>(null);

    const stopFrame: number = job?.stopFrame ?? 0;
    const jobStartFrame: number = job?.startFrame ?? 0;
    const totalFrames = Math.max(stopFrame - jobStartFrame + 1, 1);

    // Load intervals whenever the job changes; cancel on unmount to avoid setState on dead component
    useEffect(() => {
        if (!job) return undefined;
        let cancelled = false;
        setLoading(true);
        serverProxy.annotations
            .getAnnotations('job', job.id)
            .then((collection) => {
                if (!cancelled) setIntervals(collection.intervals ?? []);
            })
            .catch((err: unknown) => {
                if (!cancelled) notification.error({ message: 'Failed to load intervals', description: String(err) });
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });
        return () => { cancelled = true; };
    }, [job?.id]);

    const createInterval = useCallback(async () => {
        if (!job || selectedLabelId === null || startFrame === null || endFrame === null) return;
        if (endFrame < startFrame) {
            notification.warning({ message: 'End frame must be ≥ start frame' });
            return;
        }
        const newInterval: SerializedInterval = {
            label_id: selectedLabelId,
            frame: startFrame,
            end_frame: endFrame,
            group: 0,
            source: Source.MANUAL,
            attributes: [],
        };
        setSaving(true);
        try {
            const result = await serverProxy.annotations.updateAnnotations(
                'job',
                job.id,
                {
                    version: 0,
                    tags: [],
                    shapes: [],
                    tracks: [],
                    intervals: [newInterval],
                },
                'create',
            );
            const created: SerializedInterval[] = result.intervals ?? [newInterval];
            setIntervals((prev) => [...prev, ...created]);
            setStartFrame(null);
            setEndFrame(null);
        } catch (err: unknown) {
            notification.error({ message: 'Failed to create interval', description: String(err) });
        } finally {
            setSaving(false);
        }
    }, [job, selectedLabelId, startFrame, endFrame]);

    const deleteInterval = useCallback(async (interval: SerializedInterval) => {
        if (!job) return;
        setSaving(true);
        try {
            await serverProxy.annotations.updateAnnotations(
                'job',
                job.id,
                {
                    version: 0,
                    tags: [],
                    shapes: [],
                    tracks: [],
                    intervals: [interval],
                },
                'delete',
            );
            setIntervals((prev) => prev.filter((i) => i !== interval && i.id !== interval.id));
        } catch (err: unknown) {
            notification.error({ message: 'Failed to delete interval', description: String(err) });
        } finally {
            setSaving(false);
        }
    }, [job]);

    const updateInterval = useCallback(async (
        interval: SerializedInterval,
        newFrame: number,
        newEndFrame: number,
    ) => {
        if (!job || newEndFrame < newFrame) return;
        setSaving(true);
        try {
            const updated = { ...interval, frame: newFrame, end_frame: newEndFrame };
            await serverProxy.annotations.updateAnnotations(
                'job',
                job.id,
                {
                    version: 0,
                    tags: [],
                    shapes: [],
                    tracks: [],
                    intervals: [updated],
                },
                'update',
            );
            setIntervals((prev) => prev.map((i) => (i.id === interval.id ? updated : i)));
        } catch (err: unknown) {
            notification.error({ message: 'Failed to update interval', description: String(err) });
        } finally {
            setSaving(false);
        }
    }, [job]);

    const labelMap = Object.fromEntries(labels.map((l) => [l.id, l]));
    const canCreate = !saving && selectedLabelId !== null && startFrame !== null && endFrame !== null;

    return (
        <div className='cvat-phase-track-editor'>
            {/* ── Creation Panel ── */}
            <div className='cvat-phase-track-create'>
                <Select
                    className='cvat-phase-track-label-select'
                    placeholder='Select label'
                    value={selectedLabelId ?? undefined}
                    onChange={(v: number) => setSelectedLabelId(v)}
                >
                    {labels.map((label) => (
                        <Select.Option key={label.id} value={label.id}>
                            <span
                                className='cvat-phase-track-label-swatch'
                                style={{ backgroundColor: label.color }}
                            />
                            {label.name}
                        </Select.Option>
                    ))}
                </Select>

                <div className='cvat-phase-track-frame-inputs'>
                    <div className='cvat-phase-track-frame-row'>
                        <span className='cvat-phase-track-frame-label'>Start</span>
                        <InputNumber
                            min={jobStartFrame}
                            max={stopFrame}
                            value={startFrame ?? undefined}
                            onChange={(v) => setStartFrame(v as number | null)}
                            size='small'
                        />
                        <Tooltip title={`Set to current frame (${currentFrame})`}>
                            <Button size='small' onClick={() => setStartFrame(currentFrame)}>
                                {currentFrame}
                            </Button>
                        </Tooltip>
                    </div>
                    <div className='cvat-phase-track-frame-row'>
                        <span className='cvat-phase-track-frame-label'>End</span>
                        <InputNumber
                            min={jobStartFrame}
                            max={stopFrame}
                            value={endFrame ?? undefined}
                            onChange={(v) => setEndFrame(v as number | null)}
                            size='small'
                        />
                        <Tooltip title={`Set to current frame (${currentFrame})`}>
                            <Button size='small' onClick={() => setEndFrame(currentFrame)}>
                                {currentFrame}
                            </Button>
                        </Tooltip>
                    </div>
                </div>

                <Button
                    type='primary'
                    icon={<PlusOutlined />}
                    disabled={!canCreate}
                    onClick={createInterval}
                    block
                >
                    Add Interval
                </Button>
            </div>

            {/* ── Timeline ── */}
            <div className='cvat-phase-track-timeline'>
                {intervals.map((interval, idx) => {
                    const left = ((interval.frame - jobStartFrame) / totalFrames) * 100;
                    const width = Math.max(
                        ((interval.end_frame - interval.frame + 1) / totalFrames) * 100,
                        0.5,
                    );
                    const label = labelMap[interval.label_id];
                    const isAuto = interval.source === 'auto' || interval.source === 'semi-auto';
                    return (
                        <Tooltip
                            key={interval.id ?? `tmp-${idx}`}
                            title={`${label?.name ?? 'Unknown'}: ${interval.frame}–${interval.end_frame}${isAuto ? ' (model prediction)' : ''}`}
                        >
                            <div
                                className={`cvat-phase-track-bar${isAuto ? ' cvat-phase-track-bar-auto' : ''}`}
                                style={{
                                    left: `${left}%`,
                                    width: `${width}%`,
                                    backgroundColor: label?.color ?? '#888',
                                }}
                            />
                        </Tooltip>
                    );
                })}
                <div
                    className='cvat-phase-track-cursor'
                    style={{
                        left: `${((currentFrame - jobStartFrame) / totalFrames) * 100}%`,
                    }}
                />
            </div>

            {/* ── Interval List ── */}
            {loading ? (
                <div className='cvat-phase-track-spinner'>
                    <Spin size='small' />
                </div>
            ) : (
                <div className='cvat-phase-track-list'>
                    {intervals.length === 0 ? (
                        <p className='cvat-phase-track-empty'>
                            No intervals yet. Select a label, set start and end frames, then click Add Interval.
                        </p>
                    ) : (
                        intervals.map((interval, idx) => {
                            const label = labelMap[interval.label_id];
                            const isAuto = interval.source === 'auto' || interval.source === 'semi-auto';
                            return (
                                <div
                                    key={interval.id ?? `tmp-${idx}`}
                                    className={`cvat-phase-track-list-item${isAuto ? ' cvat-phase-track-list-item-auto' : ''}`}
                                >
                                    <span
                                        className='cvat-phase-track-list-dot'
                                        style={{ backgroundColor: label?.color ?? '#888' }}
                                    />
                                    <span className='cvat-phase-track-list-name'>
                                        {label?.name ?? 'Unknown'}
                                    </span>
                                    <InputNumber
                                        className='cvat-phase-track-inline-input'
                                        size='small'
                                        min={jobStartFrame}
                                        max={interval.end_frame}
                                        value={interval.frame}
                                        disabled={saving}
                                        onChange={(v) => {
                                            if (v !== null) updateInterval(interval, v as number, interval.end_frame);
                                        }}
                                    />
                                    <span className='cvat-phase-track-list-sep'>–</span>
                                    <InputNumber
                                        className='cvat-phase-track-inline-input'
                                        size='small'
                                        min={interval.frame}
                                        max={stopFrame}
                                        value={interval.end_frame}
                                        disabled={saving}
                                        onChange={(v) => {
                                            if (v !== null) updateInterval(interval, interval.frame, v as number);
                                        }}
                                    />
                                    <Button
                                        size='small'
                                        danger
                                        icon={<DeleteOutlined />}
                                        disabled={saving}
                                        onClick={() => deleteInterval(interval)}
                                    />
                                </div>
                            );
                        })
                    )}
                </div>
            )}
        </div>
    );
}
