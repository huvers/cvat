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

type CopilotTabTarget = 'case-info' | 'narration' | 'phase-track' | 'issues' | 'copilot';

interface CopilotPanelProps {
    onOpenTab?: (tab: CopilotTabTarget) => void;
}

const PRIORITY_COLORS: Record<string, string> = {
    high: 'red',
    medium: 'orange',
    low: 'blue',
};

const PRIORITY_LABELS: Record<Suggestion['priority'], string> = {
    high: 'Urgent',
    medium: 'Next',
    low: 'Optional',
};

const PRIORITY_ORDER: Record<Suggestion['priority'], number> = {
    high: 0,
    medium: 1,
    low: 2,
};

const SUGGESTION_LABELS: Record<string, string> = {
    create_interval: 'Create phase interval',
    adjust_interval: 'Adjust phase boundary',
    review: 'Review moment',
    accept_predictions: 'Review auto predictions',
    submit_ready: 'Ready to submit',
    seek_video: 'Jump to clip',
    flag_issue: 'Flag issue',
    add_classification: 'Add classification',
};

const COPILOT_FPS = 60;

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

function formatFrameTime(frame: number, startFrame: number): string {
    const seconds = Math.max(frame - startFrame, 0) / COPILOT_FPS;
    const minutes = Math.floor(seconds / 60);
    const remainderSeconds = Math.floor(seconds % 60);
    return `${minutes}:${String(remainderSeconds).padStart(2, '0')}`;
}

function formatSuggestionTarget(action: Suggestion['action'] | undefined, startFrame: number): string | null {
    if (!action) return null;

    if (action.start_frame !== undefined && action.end_frame !== undefined) {
        return `Target ${formatFrameTime(action.start_frame, startFrame)} - ${formatFrameTime(action.end_frame, startFrame)}`;
    }

    if (action.seek_frame !== undefined) {
        return `Jump to ${formatFrameTime(action.seek_frame, startFrame)}`;
    }

    if (action.count !== undefined) {
        return `${action.count} candidate${action.count === 1 ? '' : 's'} to review`;
    }

    return null;
}

function getActionLabel(suggestion: Suggestion): string | null {
    switch (suggestion.type) {
        case 'create_interval':
            return 'Create interval';
        case 'review':
        case 'seek_video':
        case 'adjust_interval':
            return 'Jump to frame';
        case 'accept_predictions':
            return 'Open phase track';
        case 'add_classification':
            return 'Open case info';
        case 'flag_issue':
            return 'Open issues';
        default:
            return null;
    }
}

function getNavigationTarget(suggestion: Suggestion): CopilotTabTarget | null {
    switch (suggestion.type) {
        case 'accept_predictions':
            return 'phase-track';
        case 'add_classification':
            return 'case-info';
        case 'flag_issue':
            return 'issues';
        default:
            return null;
    }
}

function getSuggestionHint(suggestion: Suggestion): string | null {
    switch (suggestion.type) {
        case 'submit_ready':
            return 'Use Submit & Next in the footer when this job looks complete.';
        case 'accept_predictions':
            return 'Open the phase track to inspect auto-generated intervals before accepting them.';
        case 'add_classification':
            return 'Open Case info to record the missing procedure metadata.';
        case 'flag_issue':
            return 'Open Issues to leave a note for follow-up or edge cases.';
        default:
            return null;
    }
}

function CompletionChecklist({ completeness }: { completeness?: Record<string, boolean> }): JSX.Element | null {
    if (!completeness) return null;
    const items = [
        { key: 'has_classifications', label: 'Classified' },
        { key: 'has_narration', label: 'Narrated' },
        { key: 'has_completed_transcript', label: 'Transcribed' },
        { key: 'has_intervals', label: 'Phases annotated' },
    ];
    const completedItems = items.filter((item) => completeness[item.key]).length;

    return (
        <div className='cvat-copilot-checklist-block'>
            <div className='cvat-copilot-checklist-heading'>
                <Typography.Text strong>Readiness</Typography.Text>
                <Typography.Text type='secondary'>
                    {`${completedItems}/${items.length} complete`}
                </Typography.Text>
            </div>
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
        </div>
    );
}

export default function CopilotPanel({ onOpenTab }: CopilotPanelProps): JSX.Element {
    const dispatch = useDispatch();
    const job = useSelector((state: CombinedState) => state.annotation.job.instance) as Job | null | undefined;
    const jobID = job?.id;

    const [data, setData] = useState<CopilotResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [lastUpdatedAt, setLastUpdatedAt] = useState<number | null>(null);

    const fetchSuggestions = useCallback(async () => {
        if (!jobID) return;
        setLoading(true);
        try {
            const result = await serverProxy.jobs.getCopilotSuggestions(jobID);
            setData(result as CopilotResponse);
            setLastUpdatedAt(Date.now());
        } catch (err: unknown) {
            notification.error({ message: 'Copilot unavailable', description: String(err) });
        } finally {
            setLoading(false);
        }
    }, [jobID]);

    // Auto-fetch on mount
    useEffect(() => {
        if (jobID) fetchSuggestions();
    }, [fetchSuggestions, jobID]);

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

        const navigationTarget = getNavigationTarget(suggestion);
        if (navigationTarget) {
            onOpenTab?.(navigationTarget);
            return;
        }

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
                break;
        }
    }, [handleCreateInterval, handleSeek, onOpenTab]);

    const ctx = data?.context;
    const startFrame = job?.startFrame ?? 0;
    const suggestions = [...(data?.suggestions ?? [])].sort((left, right) =>
        PRIORITY_ORDER[left.priority] - PRIORITY_ORDER[right.priority]);
    const summaryTimestamp = lastUpdatedAt
        ? new Intl.DateTimeFormat([], { hour: 'numeric', minute: '2-digit' }).format(lastUpdatedAt)
        : null;

    return (
        <div className='cvat-copilot-panel'>
            <div className='cvat-copilot-top'>
                <div className='cvat-copilot-header'>
                    <div className='cvat-copilot-heading'>
                        <Typography.Text strong>
                            <RobotOutlined />
                            {' Copilot'}
                        </Typography.Text>
                        <Typography.Text type='secondary'>
                            Actionable guidance for this case
                        </Typography.Text>
                    </div>
                    <Button
                        size='small'
                        icon={<ReloadOutlined spin={loading} />}
                        onClick={fetchSuggestions}
                        loading={loading}
                    >
                        Refresh
                    </Button>
                </div>

                <div className='cvat-copilot-summary-card'>
                    <div className='cvat-copilot-summary-row'>
                        <Typography.Text strong>Next best actions</Typography.Text>
                        <Typography.Text type='secondary'>
                            {loading ? 'Refreshing...' : summaryTimestamp ? `Updated ${summaryTimestamp}` : 'Not yet analyzed'}
                        </Typography.Text>
                    </div>
                    {data?.summary ? (
                        <Typography.Paragraph className='cvat-copilot-summary'>
                            {data.summary}
                        </Typography.Paragraph>
                    ) : (
                        <Typography.Paragraph className='cvat-copilot-summary'>
                            Ask Copilot for the next high-impact annotation steps in this job.
                        </Typography.Paragraph>
                    )}
                </div>

                {ctx && (
                    <div className='cvat-copilot-overview'>
                        <div className='cvat-copilot-overview-header'>
                            <div>
                                <Typography.Text strong>Job snapshot</Typography.Text>
                                <Typography.Text type='secondary'>
                                    {ctx.procedure_type}
                                </Typography.Text>
                            </div>
                            <Tag color={ctx.coverage_pct >= 90 ? 'success' : ctx.coverage_pct >= 50 ? 'warning' : 'error'}>
                                {`${ctx.coverage_pct}% covered`}
                            </Tag>
                        </div>
                        <div className='cvat-copilot-metric-grid'>
                            <div className='cvat-copilot-metric'>
                                <span className='cvat-copilot-metric-value'>{ctx.duration}</span>
                                <span className='cvat-copilot-metric-label'>Duration</span>
                            </div>
                            <div className='cvat-copilot-metric'>
                                <span className='cvat-copilot-metric-value'>{ctx.interval_count}</span>
                                <span className='cvat-copilot-metric-label'>Intervals</span>
                            </div>
                            <div className='cvat-copilot-metric'>
                                <span className='cvat-copilot-metric-value'>{ctx.gap_count}</span>
                                <span className='cvat-copilot-metric-label'>Coverage gaps</span>
                            </div>
                            <div className='cvat-copilot-metric'>
                                <span className='cvat-copilot-metric-value'>{ctx.auto_count}</span>
                                <span className='cvat-copilot-metric-label'>Auto intervals</span>
                            </div>
                        </div>
                        <div className='cvat-copilot-coverage-track'>
                            <div
                                className='cvat-copilot-coverage-fill'
                                style={{ width: `${Math.max(0, Math.min(ctx.coverage_pct, 100))}%` }}
                            />
                        </div>
                        <CompletionChecklist completeness={ctx.completeness} />
                    </div>
                )}
            </div>

            {loading && !data && (
                <div className='cvat-copilot-spinner'>
                    <Spin size='small' />
                    <Typography.Text type='secondary'>Analyzing job...</Typography.Text>
                </div>
            )}

            {data?.error && (
                <div className='cvat-copilot-state cvat-copilot-state-error'>
                    <WarningOutlined />
                    <div>
                        <Typography.Text strong>Copilot is unavailable</Typography.Text>
                        <Typography.Paragraph>
                            {data.summary}
                        </Typography.Paragraph>
                    </div>
                </div>
            )}

            {!data?.error && suggestions.length > 0 && (
                <>
                    <div className='cvat-copilot-section-header'>
                        <Typography.Text strong>Suggested next steps</Typography.Text>
                        <Typography.Text type='secondary'>
                            {`${suggestions.length} ${suggestions.length === 1 ? 'item' : 'items'}`}
                        </Typography.Text>
                    </div>
                    <div className='cvat-copilot-suggestions'>
                        {suggestions.map((suggestion, idx) => {
                            const actionLabel = getActionLabel(suggestion);
                            const targetLabel = formatSuggestionTarget(suggestion.action, startFrame);
                            const hint = getSuggestionHint(suggestion);
                            const showAction = Boolean(actionLabel && (suggestion.action || getNavigationTarget(suggestion)));

                            return (
                                <div
                                    key={`suggestion-${idx}`}
                                    className={`cvat-copilot-suggestion-card cvat-copilot-priority-${suggestion.priority}`}
                                >
                                    <div className='cvat-copilot-suggestion-top'>
                                        <div className='cvat-copilot-suggestion-title'>
                                            <span className='cvat-copilot-suggestion-icon'>
                                                {TYPE_ICONS[suggestion.type] ?? <RobotOutlined />}
                                            </span>
                                            <div>
                                                <Typography.Text strong>
                                                    {SUGGESTION_LABELS[suggestion.type] ?? 'Suggested action'}
                                                </Typography.Text>
                                                {targetLabel && (
                                                    <Typography.Text type='secondary' className='cvat-copilot-suggestion-target'>
                                                        {targetLabel}
                                                    </Typography.Text>
                                                )}
                                            </div>
                                        </div>
                                        <Tag
                                            color={PRIORITY_COLORS[suggestion.priority] ?? 'default'}
                                            className='cvat-copilot-suggestion-priority'
                                        >
                                            {PRIORITY_LABELS[suggestion.priority] ?? suggestion.priority}
                                        </Tag>
                                    </div>
                                    <Typography.Text className='cvat-copilot-suggestion-message'>
                                        {suggestion.message}
                                    </Typography.Text>
                                    <div className='cvat-copilot-suggestion-footer'>
                                        {hint && (
                                            <Typography.Text type='secondary' className='cvat-copilot-suggestion-hint'>
                                                {hint}
                                            </Typography.Text>
                                        )}
                                        {showAction && (
                                            <Button
                                                size='small'
                                                type='primary'
                                                onClick={() => handleAction(suggestion)}
                                                className='cvat-copilot-suggestion-action'
                                            >
                                                {actionLabel}
                                            </Button>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </>
            )}

            {data && !data.error && suggestions.length === 0 && (
                <div className='cvat-copilot-complete'>
                    <CheckCircleOutlined style={{ fontSize: 24, color: '#52c41a' }} />
                    <Typography.Text strong>Nothing urgent right now</Typography.Text>
                    <Typography.Text type='secondary'>
                        Copilot does not see any immediate follow-up actions for this job.
                    </Typography.Text>
                </div>
            )}
        </div>
    );
}
