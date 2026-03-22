// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useEffect, useCallback } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import Button from 'antd/lib/button';
import InputNumber from 'antd/lib/input-number';
import Select from 'antd/lib/select';
import Spin from 'antd/lib/spin';
import Tooltip from 'antd/lib/tooltip';
import notification from 'antd/lib/notification';
import { CheckOutlined, DeleteOutlined, PlusOutlined } from '@ant-design/icons';

import { Job } from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import { Source } from 'cvat-core/src/enums';
import { ShortcutScope } from 'utils/enums';
import { registerComponentShortcuts } from 'actions/shortcuts-actions';
import { subKeyMap } from 'utils/component-subkeymap';
import GlobalHotKeys from 'utils/mousetrap-react';
import { changeFrameAsync } from 'actions/annotation-actions';
import serverProxy from 'cvat-core/src/server-proxy';
import { SerializedInterval } from 'cvat-core/src/server-response-types';

import { frameToTime } from '../utils';

import './styles.scss';

const componentShortcuts = {
    SET_INTERVAL_START: {
        name: 'Set interval start',
        description: 'Set the interval start frame to the current frame',
        sequences: ['['],
        scope: ShortcutScope.ANNOTATION_PAGE,
    },
    SET_INTERVAL_END: {
        name: 'Set interval end',
        description: 'Set the interval end frame to the current frame',
        sequences: [']'],
        scope: ShortcutScope.ANNOTATION_PAGE,
    },
    CREATE_INTERVAL: {
        name: 'Create interval',
        description: 'Create a new interval with the selected label and frame range',
        sequences: ['enter'],
        scope: ShortcutScope.ANNOTATION_PAGE,
    },
    ACCEPT_ALL_PREDICTIONS: {
        name: 'Accept all predictions',
        description: 'Convert all auto-predicted intervals to manual',
        sequences: ['ctrl+shift+a'],
        scope: ShortcutScope.ANNOTATION_PAGE,
    },
};

registerComponentShortcuts(componentShortcuts);

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

    const dispatch = useDispatch();

    const seekToFrame = useCallback((frame: number) => {
        dispatch(changeFrameAsync(frame));
    }, [dispatch]);

    const acceptAllAuto = useCallback(async () => {
        if (!job) return;
        const autoIntervals = intervals.filter(
            (i) => i.source === 'auto' || i.source === 'semi-auto',
        );
        if (autoIntervals.length === 0) return;
        setSaving(true);
        try {
            const updated = autoIntervals.map((i) => ({ ...i, source: Source.MANUAL }));
            await serverProxy.annotations.updateAnnotations(
                'job',
                job.id,
                { version: 0, tags: [], shapes: [], tracks: [], intervals: updated },
                'update',
            );
            setIntervals((prev) => prev.map((i) => {
                const match = updated.find((u) => u.id === i.id);
                return match ?? i;
            }));
            notification.success({ message: `Accepted ${autoIntervals.length} predictions` });
        } catch (err: unknown) {
            notification.error({ message: 'Failed to accept predictions', description: String(err) });
        } finally {
            setSaving(false);
        }
    }, [job, intervals]);

    const autoCount = intervals.filter((i) => i.source === 'auto' || i.source === 'semi-auto').length;

    const { keyMap } = useSelector((state: CombinedState) => state.shortcuts);

    const labelMap = Object.fromEntries(labels.map((l) => [l.id, l]));
    const canCreate = !saving && selectedLabelId !== null && startFrame !== null && endFrame !== null;

    const shortcutHandlers: Record<keyof typeof componentShortcuts, (event?: KeyboardEvent) => void> = {
        SET_INTERVAL_START: (event) => {
            event?.preventDefault();
            setStartFrame(currentFrame);
            notification.info({ message: `Start: ${frameToTime(currentFrame, jobStartFrame)}`, duration: 1 });
        },
        SET_INTERVAL_END: (event) => {
            event?.preventDefault();
            setEndFrame(currentFrame);
            notification.info({ message: `End: ${frameToTime(currentFrame, jobStartFrame)}`, duration: 1 });
        },
        CREATE_INTERVAL: (event) => {
            event?.preventDefault();
            if (canCreate) createInterval();
        },
        ACCEPT_ALL_PREDICTIONS: (event) => {
            event?.preventDefault();
            if (autoCount > 0) acceptAllAuto();
        },
    };

    return (
        <div className='cvat-phase-track-editor'>
            <GlobalHotKeys
                keyMap={subKeyMap(componentShortcuts, keyMap)}
                handlers={shortcutHandlers}
            />
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
                        <Tooltip title={`Set to current frame [ ${frameToTime(currentFrame, jobStartFrame)}`}>
                            <Button size='small' onClick={() => setStartFrame(currentFrame)}>
                                {`[ ${frameToTime(currentFrame, jobStartFrame)}`}
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
                        <Tooltip title={`Set to current frame ] ${frameToTime(currentFrame, jobStartFrame)}`}>
                            <Button size='small' onClick={() => setEndFrame(currentFrame)}>
                                {`] ${frameToTime(currentFrame, jobStartFrame)}`}
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
                    Add Interval (Enter)
                </Button>
                {autoCount > 0 && (
                    <Button
                        icon={<CheckOutlined />}
                        disabled={saving}
                        onClick={acceptAllAuto}
                        block
                    >
                        {`Accept ${autoCount} prediction${autoCount > 1 ? 's' : ''}`}
                    </Button>
                )}
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
                    const timeRange = `${frameToTime(interval.frame, jobStartFrame)}–${frameToTime(interval.end_frame, jobStartFrame)}`;
                    return (
                        <Tooltip
                            key={interval.id ?? `tmp-${idx}`}
                            title={`${label?.name ?? 'Unknown'}: ${timeRange}${isAuto ? ' (model prediction)' : ''}`}
                        >
                            <div
                                className={`cvat-phase-track-bar${isAuto ? ' cvat-phase-track-bar-auto' : ''}`}
                                style={{
                                    left: `${left}%`,
                                    width: `${width}%`,
                                    backgroundColor: label?.color ?? '#888',
                                    cursor: 'pointer',
                                }}
                                onClick={() => seekToFrame(interval.frame)}
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
            {/* ── Time scale ── */}
            <div className='cvat-phase-track-timescale'>
                {Array.from({ length: 7 }, (_, i) => {
                    const pct = (i / 6) * 100;
                    const frame = jobStartFrame + Math.round((i / 6) * totalFrames);
                    return (
                        <span key={i} className='cvat-phase-track-tick' style={{ left: `${pct}%` }}>
                            {frameToTime(frame, jobStartFrame)}
                        </span>
                    );
                })}
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
                                    <span
                                        className='cvat-phase-track-list-name'
                                        onClick={() => seekToFrame(interval.frame)}
                                        style={{ cursor: 'pointer' }}
                                    >
                                        {label?.name ?? 'Unknown'}
                                    </span>
                                    <span className='cvat-phase-track-list-time'>
                                        {frameToTime(interval.frame, jobStartFrame)}
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
                                    <span className='cvat-phase-track-list-time'>
                                        {frameToTime(interval.end_frame, jobStartFrame)}
                                    </span>
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
