// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useEffect, useCallback } from 'react';
import { useSelector } from 'react-redux';
import Button from 'antd/lib/button';
import Collapse from 'antd/lib/collapse';
import Input from 'antd/lib/input';
import Spin from 'antd/lib/spin';
import Tag from 'antd/lib/tag';
import Tooltip from 'antd/lib/tooltip';
import Typography from 'antd/lib/typography';
import notification from 'antd/lib/notification';
import {
    SyncOutlined, CheckCircleOutlined, CloseCircleOutlined,
    ClockCircleOutlined, EditOutlined, SaveOutlined, CloseOutlined,
    FieldTimeOutlined,
} from '@ant-design/icons';

import { Job } from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import { Source } from 'cvat-core/src/enums';
import serverProxy from 'cvat-core/src/server-proxy';
import { SerializedJobTranscript, TranscriptStatus } from 'cvat-core/src/server-response-types';

import './styles.scss';

const STATUS_CONFIG: Record<TranscriptStatus, { color: string; icon: JSX.Element; label: string }> = {
    pending: { color: 'default', icon: <ClockCircleOutlined />, label: 'Pending' },
    processing: { color: 'processing', icon: <SyncOutlined spin />, label: 'Processing' },
    completed: { color: 'success', icon: <CheckCircleOutlined />, label: 'Completed' },
    failed: { color: 'error', icon: <CloseCircleOutlined />, label: 'Failed' },
};

const POLL_INTERVAL_MS = 5000;

function WordTimestampBar(props: {
    transcript: SerializedJobTranscript;
    jobStartFrame: number;
    fps: number;
}): JSX.Element | null {
    const { transcript, jobStartFrame, fps } = props;

    const words = transcript.word_timestamps;
    if (!words || words.length === 0) return null;

    const handleCreateInterval = useCallback((): void => {
        if (words.length === 0) return;
        const startSec = words[0].start;
        const endSec = words[words.length - 1].end;
        const startFrame = jobStartFrame + Math.round(startSec * fps);
        const endFrame = jobStartFrame + Math.round(endSec * fps);

        notification.info({
            message: 'Create interval from transcript',
            description: `Frames ${startFrame}–${endFrame} (${words.length} words, ${startSec.toFixed(1)}s–${endSec.toFixed(1)}s). Use the Phase track tab to create an interval with these frame bounds.`,
            duration: 8,
        });
    }, [words, jobStartFrame, fps]);

    return (
        <div className='cvat-transcript-word-bar'>
            <Tooltip title='Suggest interval from word timestamps'>
                <Button
                    size='small'
                    icon={<FieldTimeOutlined />}
                    onClick={handleCreateInterval}
                >
                    {`${words.length} words · ${words[0]?.start.toFixed(1)}s–${words[words.length - 1]?.end.toFixed(1)}s`}
                </Button>
            </Tooltip>
        </div>
    );
}

interface TranscriptViewerProps {
    refreshKey?: number;
}

export default function TranscriptViewer(props: TranscriptViewerProps): JSX.Element {
    const { refreshKey } = props;
    const job = useSelector((state: CombinedState) => state.annotation.job.instance) as Job | null | undefined;

    const [transcripts, setTranscripts] = useState<SerializedJobTranscript[]>([]);
    const [loading, setLoading] = useState(false);
    const [editingId, setEditingId] = useState<number | null>(null);
    const [editText, setEditText] = useState('');
    const [saving, setSaving] = useState(false);

    const jobStartFrame = job?.startFrame ?? 0;
    // Estimate FPS from job metadata (default 30)
    const fps = 30;

    // Load transcripts and poll while any are pending/processing
    useEffect(() => {
        if (!job) return undefined;
        let cancelled = false;
        let timer: ReturnType<typeof setTimeout> | null = null;

        const fetchTranscripts = (): void => {
            serverProxy.jobs
                .getTranscripts(job.id)
                .then((data) => {
                    if (cancelled) return;
                    setTranscripts(data);
                    setLoading(false);

                    const hasPending = data.some(
                        (t) => t.status === 'pending' || t.status === 'processing',
                    );
                    if (hasPending) {
                        timer = setTimeout(fetchTranscripts, POLL_INTERVAL_MS);
                    }
                })
                .catch((err: unknown) => {
                    if (!cancelled) {
                        setLoading(false);
                        notification.error({
                            message: 'Failed to load transcripts',
                            description: String(err),
                        });
                    }
                });
        };

        setLoading(true);
        fetchTranscripts();

        return () => {
            cancelled = true;
            if (timer) clearTimeout(timer);
        };
    }, [job?.id, refreshKey]);

    const startEditing = useCallback((t: SerializedJobTranscript) => {
        setEditingId(t.id);
        setEditText(t.corrected_transcript);
    }, []);

    const cancelEditing = useCallback(() => {
        setEditingId(null);
        setEditText('');
    }, []);

    const saveEditing = useCallback(async () => {
        if (!job || editingId === null) return;
        setSaving(true);
        try {
            const updated = await serverProxy.jobs.updateTranscript(job.id, editingId, editText);
            setTranscripts((prev) => prev.map((t) => (t.id === editingId ? updated : t)));
            setEditingId(null);
            setEditText('');
            notification.success({ message: 'Transcript saved' });
        } catch (err: unknown) {
            notification.error({ message: 'Failed to save transcript', description: String(err) });
        } finally {
            setSaving(false);
        }
    }, [job, editingId, editText]);

    if (loading && transcripts.length === 0) {
        return (
            <div className='cvat-transcript-spinner'>
                <Spin size='small' />
            </div>
        );
    }

    if (transcripts.length === 0) {
        return (
            <Typography.Paragraph className='cvat-transcript-empty'>
                No transcripts yet. Record and upload a narration above to start the transcription pipeline.
            </Typography.Paragraph>
        );
    }

    return (
        <div className='cvat-transcript-viewer'>
            <Collapse
                accordion
                defaultActiveKey={transcripts[0]?.id}
                items={transcripts.map((t) => {
                    const cfg = STATUS_CONFIG[t.status];
                    const isEditing = editingId === t.id;
                    return {
                        key: t.id,
                        label: (
                            <span className='cvat-transcript-header'>
                                <span>{`Narration #${t.narration_id}`}</span>
                                <Tag icon={cfg.icon} color={cfg.color}>{cfg.label}</Tag>
                            </span>
                        ),
                        children: (
                            <div className='cvat-transcript-content'>
                                {t.status === 'completed' && (
                                    <>
                                        <div className='cvat-transcript-section'>
                                            <div className='cvat-transcript-section-header'>
                                                <Typography.Text strong>Corrected transcript</Typography.Text>
                                                {!isEditing ? (
                                                    <Button
                                                        size='small'
                                                        icon={<EditOutlined />}
                                                        onClick={() => startEditing(t)}
                                                    >
                                                        Edit
                                                    </Button>
                                                ) : (
                                                    <span className='cvat-transcript-edit-actions'>
                                                        <Button
                                                            size='small'
                                                            type='primary'
                                                            icon={<SaveOutlined />}
                                                            loading={saving}
                                                            onClick={saveEditing}
                                                        >
                                                            Save
                                                        </Button>
                                                        <Button
                                                            size='small'
                                                            icon={<CloseOutlined />}
                                                            disabled={saving}
                                                            onClick={cancelEditing}
                                                        >
                                                            Cancel
                                                        </Button>
                                                    </span>
                                                )}
                                            </div>
                                            {isEditing ? (
                                                <Input.TextArea
                                                    className='cvat-transcript-edit-area'
                                                    value={editText}
                                                    onChange={(e) => setEditText(e.target.value)}
                                                    autoSize={{ minRows: 3, maxRows: 12 }}
                                                />
                                            ) : (
                                                <Typography.Paragraph className='cvat-transcript-text'>
                                                    {t.corrected_transcript}
                                                </Typography.Paragraph>
                                            )}
                                        </div>
                                        <WordTimestampBar
                                            transcript={t}
                                            jobStartFrame={jobStartFrame}
                                            fps={fps}
                                        />
                                        <Collapse
                                            size='small'
                                            items={[{
                                                key: 'raw',
                                                label: 'Raw ASR output',
                                                children: (
                                                    <Typography.Paragraph
                                                        className='cvat-transcript-text cvat-transcript-raw'
                                                    >
                                                        {t.raw_transcript}
                                                    </Typography.Paragraph>
                                                ),
                                            }]}
                                        />
                                        {t.model_info?.procedure_context && (
                                            <Typography.Text
                                                type='secondary'
                                                className='cvat-transcript-context'
                                            >
                                                {t.model_info.procedure_context}
                                            </Typography.Text>
                                        )}
                                    </>
                                )}
                                {t.status === 'failed' && (
                                    <Typography.Text type='danger'>
                                        {t.error_message || 'Transcription failed. Check server logs.'}
                                    </Typography.Text>
                                )}
                                {(t.status === 'pending' || t.status === 'processing') && (
                                    <div className='cvat-transcript-processing'>
                                        <Spin size='small' />
                                        <Typography.Text type='secondary'>
                                            {t.status === 'pending'
                                                ? 'Queued for transcription...'
                                                : 'Parakeet ASR + LLM correction in progress...'}
                                        </Typography.Text>
                                    </div>
                                )}
                            </div>
                        ),
                    };
                })}
            />
        </div>
    );
}
