// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';
import Badge from 'antd/lib/badge';
import Collapse from 'antd/lib/collapse';
import Spin from 'antd/lib/spin';
import Tag from 'antd/lib/tag';
import Typography from 'antd/lib/typography';
import notification from 'antd/lib/notification';
import { SyncOutlined, CheckCircleOutlined, CloseCircleOutlined, ClockCircleOutlined } from '@ant-design/icons';

import { Job } from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
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

export default function TranscriptViewer(): JSX.Element {
    const job = useSelector((state: CombinedState) => state.annotation.job.instance) as Job | null | undefined;

    const [transcripts, setTranscripts] = useState<SerializedJobTranscript[]>([]);
    const [loading, setLoading] = useState(false);

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
    }, [job?.id]);

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
                No transcripts yet. Record and upload a narration to start the transcription pipeline.
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
                                            <Typography.Text strong>Corrected transcript</Typography.Text>
                                            <Typography.Paragraph className='cvat-transcript-text'>
                                                {t.corrected_transcript}
                                            </Typography.Paragraph>
                                        </div>
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
