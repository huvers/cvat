// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useCallback, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import Button from 'antd/lib/button';
import Spin from 'antd/lib/spin';
import Tag from 'antd/lib/tag';
import Typography from 'antd/lib/typography';
import notification from 'antd/lib/notification';
import {
    RobotOutlined, CheckCircleOutlined, WarningOutlined,
    PlusOutlined, AimOutlined, ThunderboltOutlined,
    ReloadOutlined, ClockCircleOutlined,
} from '@ant-design/icons';

import { Job } from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import { Source } from 'cvat-core/src/enums';
import { changeFrameAsync } from 'actions/annotation-actions';
import serverProxy from 'cvat-core/src/server-proxy';

import './styles.scss';

interface Suggestion {
    type: string;
    priority: 'high' | 'medium' | 'low';
    message: string;
    action?: {
        label?: string;
        start_frame?: number;
        end_frame?: number;
        seek_frame?: number;
        count?: number;
    };
}

interface CopilotResponse {
    suggestions: Suggestion[];
    summary: string;
    error?: boolean;
    context?: {
        procedure_type: string;
        coverage_pct: number;
        gap_count: number;
        interval_count: number;
        auto_count: number;
        completeness: Record<string, boolean>;
        duration: string;
    };
}

const PRIORITY_COLORS: Record<string, string> = {
    high: 'red',
    medium: 'orange',
    low: 'blue',
};

const TYPE_ICONS: Record<string, JSX.Element> = {
    create_interval: <PlusOutlined />,
    adjust_interval: <AimOutlined />,
    review: <WarningOutlined />,
    accept_predictions: <ThunderboltOutlined />,
    submit_ready: <CheckCircleOutlined />,
    seek_video: <AimOutlined />,
    flag_issue: <WarningOutlined />,
    add_classification: <PlusOutlined />,
};

function CompletionChecklist({ completeness }: { completeness?: Record<string, boolean> }): JSX.Element | null {
    if (!completeness) return null;
    const items = [
        { key: 'has_classifications', label: 'Classified' },
        { key: 'has_narration', label: 'Narrated' },
        { key: 'has_completed_transcript', label: 'Transcribed' },
        { key: 'has_intervals', label: 'Phases annotated' },
    ];
    return (
        <div className='cvat-copilot-checklist'>
            {items.map((item) => (
                <span key={item.key} className='cvat-copilot-checklist-item'>
                    {completeness[item.key]
                        ? <CheckCircleOutlined style={{ color: '#52c41a' }} />
                        : <ClockCircleOutlined style={{ color: '#d9d9d9' }} />}
                    <span className={completeness[item.key] ? '' : 'cvat-copilot-checklist-pending'}>
                        {item.label}
                    </span>
                </span>
            ))}
        </div>
    );
}

export default function CopilotPanel(): JSX.Element {
    const dispatch = useDispatch();
    const job = useSelector((state: CombinedState) => state.annotation.job.instance) as Job | null | undefined;

    const [data, setData] = useState<CopilotResponse | null>(null);
    const [loading, setLoading] = useState(false);

    const fetchSuggestions = useCallback(async () => {
        if (!job) return;
        setLoading(true);
        try {
            const result = await serverProxy.jobs.getCopilotSuggestions(job.id);
            setData(result as CopilotResponse);
        } catch (err: unknown) {
            notification.error({ message: 'Copilot unavailable', description: String(err) });
        } finally {
            setLoading(false);
        }
    }, [job]);

    // Auto-fetch on mount
    useEffect(() => {
        if (job) fetchSuggestions();
    }, [job?.id]);

    const handleSeek = useCallback((frame: number) => {
        dispatch(changeFrameAsync(frame));
    }, [dispatch]);

    const handleCreateInterval = useCallback(async (action: Suggestion['action']) => {
        if (!job || !action?.label || action.start_frame === undefined || action.end_frame === undefined) return;

        // Find the label by name
        const labels = job.labels || [];
        const label = labels.find((l: any) => l.name === action.label);
        if (!label) {
            notification.warning({ message: `Label "${action.label}" not found in this job` });
            return;
        }

        try {
            await serverProxy.annotations.updateAnnotations(
                'job',
                job.id,
                {
                    version: 0,
                    tags: [],
                    shapes: [],
                    tracks: [],
                    intervals: [{
                        label_id: label.id!,
                        frame: action.start_frame!,
                        end_frame: action.end_frame!,
                        group: 0,
                        source: Source.MANUAL,
                        attributes: [],
                    }],
                },
                'create',
            );
            notification.success({ message: `Created interval: ${action.label}` });
            // Refresh suggestions after action
            fetchSuggestions();
        } catch (err: unknown) {
            notification.error({ message: 'Failed to create interval', description: String(err) });
        }
    }, [job, fetchSuggestions]);

    const handleAction = useCallback((suggestion: Suggestion) => {
        const { action } = suggestion;
        if (!action) return;

        switch (suggestion.type) {
            case 'create_interval':
                handleCreateInterval(action);
                break;
            case 'review':
            case 'seek_video':
            case 'adjust_interval':
                if (action.seek_frame !== undefined) handleSeek(action.seek_frame);
                else if (action.start_frame !== undefined) handleSeek(action.start_frame);
                break;
            default:
                notification.info({ message: suggestion.message });
                break;
        }
    }, [handleCreateInterval, handleSeek]);

    const ctx = data?.context;

    return (
        <div className='cvat-copilot-panel'>
            {/* Progress overview */}
            {ctx && (
                <div className='cvat-copilot-overview'>
                    <div className='cvat-copilot-overview-stats'>
                        <Tag>{ctx.procedure_type}</Tag>
                        <Tag>{ctx.duration}</Tag>
                        <Tag color={ctx.coverage_pct >= 90 ? 'success' : ctx.coverage_pct >= 50 ? 'warning' : 'error'}>
                            {`${ctx.coverage_pct}% covered`}
                        </Tag>
                        <Tag>{`${ctx.interval_count} phases`}</Tag>
                        {ctx.auto_count > 0 && <Tag color='blue'>{`${ctx.auto_count} auto`}</Tag>}
                    </div>
                    <CompletionChecklist completeness={ctx.completeness} />
                </div>
            )}

            {/* Suggestions */}
            <div className='cvat-copilot-header'>
                <Typography.Text strong>
                    <RobotOutlined />
                    {' Copilot'}
                </Typography.Text>
                <Button
                    size='small'
                    icon={<ReloadOutlined spin={loading} />}
                    onClick={fetchSuggestions}
                    loading={loading}
                >
                    Refresh
                </Button>
            </div>

            {loading && !data && (
                <div className='cvat-copilot-spinner'>
                    <Spin size='small' />
                    <Typography.Text type='secondary'>Analyzing job...</Typography.Text>
                </div>
            )}

            {data?.summary && (
                <Typography.Paragraph className='cvat-copilot-summary'>
                    {data.summary}
                </Typography.Paragraph>
            )}

            {data?.suggestions && data.suggestions.length > 0 && (
                <div className='cvat-copilot-suggestions'>
                    {data.suggestions.map((suggestion, idx) => (
                        <div
                            key={`suggestion-${idx}`}
                            className='cvat-copilot-suggestion-card'
                        >
                            <div className='cvat-copilot-suggestion-top'>
                                <span className='cvat-copilot-suggestion-icon'>
                                    {TYPE_ICONS[suggestion.type] ?? <RobotOutlined />}
                                </span>
                                <Tag
                                    color={PRIORITY_COLORS[suggestion.priority] ?? 'default'}
                                    className='cvat-copilot-suggestion-priority'
                                >
                                    {suggestion.priority}
                                </Tag>
                            </div>
                            <Typography.Text className='cvat-copilot-suggestion-message'>
                                {suggestion.message}
                            </Typography.Text>
                            {suggestion.action && (
                                <Button
                                    size='small'
                                    type='primary'
                                    onClick={() => handleAction(suggestion)}
                                    className='cvat-copilot-suggestion-action'
                                >
                                    {suggestion.type === 'create_interval' ? 'Create' :
                                        suggestion.type === 'accept_predictions' ? 'Accept All' :
                                            suggestion.type === 'submit_ready' ? 'Submit' :
                                                'Go'}
                                </Button>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {data && !data.error && data.suggestions?.length === 0 && (
                <div className='cvat-copilot-complete'>
                    <CheckCircleOutlined style={{ fontSize: 24, color: '#52c41a' }} />
                    <Typography.Text>Looking good! No suggestions right now.</Typography.Text>
                </div>
            )}
        </div>
    );
}
