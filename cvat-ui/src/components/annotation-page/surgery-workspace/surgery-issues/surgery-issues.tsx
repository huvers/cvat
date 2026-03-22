// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useEffect, useCallback } from 'react';
import { useSelector } from 'react-redux';
import Button from 'antd/lib/button';
import Input from 'antd/lib/input';
import InputNumber from 'antd/lib/input-number';
import Select from 'antd/lib/select';
import Spin from 'antd/lib/spin';
import Tag from 'antd/lib/tag';
import Tooltip from 'antd/lib/tooltip';
import Typography from 'antd/lib/typography';
import notification from 'antd/lib/notification';
import {
    PlusOutlined, CheckOutlined, ExclamationCircleOutlined,
    ClockCircleOutlined,
} from '@ant-design/icons';

import { Job } from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import serverProxy from 'cvat-core/src/server-proxy';

import './styles.scss';

interface SurgeryIssue {
    id: number;
    frame: number;
    end_frame: number | null;
    issue_type: 'frame' | 'interval' | 'narration';
    resolved: boolean;
    owner: { username: string } | null;
    comments: { count: number };
    created_date: string;
}

const TYPE_COLORS: Record<string, string> = {
    frame: 'blue',
    interval: 'orange',
    narration: 'purple',
};

export default function SurgeryIssues(): JSX.Element {
    const job = useSelector((state: CombinedState) => state.annotation.job.instance) as Job | null | undefined;
    const currentFrame = useSelector((state: CombinedState) => state.annotation.player.frame.number);

    const [issues, setIssues] = useState<SurgeryIssue[]>([]);
    const [loading, setLoading] = useState(false);
    const [creating, setCreating] = useState(false);

    // New issue form
    const [showForm, setShowForm] = useState(false);
    const [issueType, setIssueType] = useState<'frame' | 'interval' | 'narration'>('interval');
    const [startFrame, setStartFrame] = useState<number | null>(null);
    const [endFrame, setEndFrame] = useState<number | null>(null);
    const [message, setMessage] = useState('');

    useEffect(() => {
        if (!job) return undefined;
        let cancelled = false;
        setLoading(true);
        serverProxy.issues
            .get({ job_id: job.id })
            .then((data: any) => {
                if (!cancelled) setIssues(Array.isArray(data) ? data : []);
            })
            .catch((err: unknown) => {
                if (!cancelled) notification.error({ message: 'Failed to load issues', description: String(err) });
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });
        return () => { cancelled = true; };
    }, [job?.id]);

    const createIssue = useCallback(async () => {
        if (!job || !message.trim()) return;
        setCreating(true);
        try {
            const payload: Record<string, any> = {
                job: job.id,
                frame: startFrame ?? currentFrame,
                position: [0, 0, 0, 0],
                message: message.trim(),
                issue_type: issueType,
            };
            if (issueType === 'interval' && endFrame !== null) {
                payload.end_frame = endFrame;
            }
            await serverProxy.issues.create(payload);
            // Refresh
            const data: any = await serverProxy.issues.get({ job_id: job.id });
            setIssues(Array.isArray(data) ? data : []);
            setShowForm(false);
            setMessage('');
            setStartFrame(null);
            setEndFrame(null);
            notification.success({ message: 'Issue created' });
        } catch (err: unknown) {
            notification.error({ message: 'Failed to create issue', description: String(err) });
        } finally {
            setCreating(false);
        }
    }, [job, message, issueType, startFrame, endFrame, currentFrame]);

    const resolveIssue = useCallback(async (issueId: number) => {
        try {
            await serverProxy.issues.update(issueId, { resolved: true });
            setIssues((prev) => prev.map((i) => (i.id === issueId ? { ...i, resolved: true } : i)));
        } catch (err: unknown) {
            notification.error({ message: 'Failed to resolve issue', description: String(err) });
        }
    }, []);

    const openCount = issues.filter((i) => !i.resolved).length;

    return (
        <div className='cvat-surgery-issues'>
            <div className='cvat-surgery-issues-header'>
                <Typography.Text strong>
                    {`Issues (${openCount} open)`}
                </Typography.Text>
                <Button
                    size='small'
                    icon={<PlusOutlined />}
                    onClick={() => setShowForm(!showForm)}
                >
                    New
                </Button>
            </div>

            {showForm && (
                <div className='cvat-surgery-issues-form'>
                    <Select
                        size='small'
                        value={issueType}
                        onChange={(v) => setIssueType(v)}
                        style={{ width: '100%' }}
                    >
                        <Select.Option value='interval'>Interval (frame range)</Select.Option>
                        <Select.Option value='frame'>Frame (single)</Select.Option>
                        <Select.Option value='narration'>Narration</Select.Option>
                    </Select>
                    {issueType !== 'narration' && (
                        <div className='cvat-surgery-issues-frame-row'>
                            <InputNumber
                                size='small'
                                placeholder='Start'
                                value={startFrame ?? undefined}
                                onChange={(v) => setStartFrame(v as number | null)}
                            />
                            {issueType === 'interval' && (
                                <>
                                    <span>–</span>
                                    <InputNumber
                                        size='small'
                                        placeholder='End'
                                        value={endFrame ?? undefined}
                                        onChange={(v) => setEndFrame(v as number | null)}
                                    />
                                </>
                            )}
                            <Button size='small' onClick={() => setStartFrame(currentFrame)}>
                                Current
                            </Button>
                        </div>
                    )}
                    <Input.TextArea
                        size='small'
                        placeholder='Describe the issue...'
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        autoSize={{ minRows: 2, maxRows: 4 }}
                    />
                    <Button
                        type='primary'
                        size='small'
                        loading={creating}
                        disabled={!message.trim()}
                        onClick={createIssue}
                        block
                    >
                        Create Issue
                    </Button>
                </div>
            )}

            {loading ? (
                <div className='cvat-surgery-issues-spinner'>
                    <Spin size='small' />
                </div>
            ) : issues.length === 0 ? (
                <Typography.Paragraph className='cvat-surgery-issues-empty'>
                    No issues. Click New to report a phase boundary error, narration mismatch, or other concern.
                </Typography.Paragraph>
            ) : (
                <div className='cvat-surgery-issues-list'>
                    {issues.map((issue) => (
                        <div
                            key={issue.id}
                            className={`cvat-surgery-issues-item${issue.resolved ? ' cvat-surgery-issues-item-resolved' : ''}`}
                        >
                            <div className='cvat-surgery-issues-item-top'>
                                <Tag color={TYPE_COLORS[issue.issue_type] ?? 'default'}>
                                    {issue.issue_type}
                                </Tag>
                                <span className='cvat-surgery-issues-item-frames'>
                                    {issue.end_frame !== null
                                        ? `${issue.frame}–${issue.end_frame}`
                                        : `Frame ${issue.frame}`}
                                </span>
                                {issue.resolved ? (
                                    <Tag icon={<CheckOutlined />} color='success'>Resolved</Tag>
                                ) : (
                                    <Tooltip title='Resolve'>
                                        <Button
                                            size='small'
                                            type='text'
                                            icon={<CheckOutlined />}
                                            onClick={() => resolveIssue(issue.id)}
                                        />
                                    </Tooltip>
                                )}
                            </div>
                            <span className='cvat-surgery-issues-item-meta'>
                                {issue.owner?.username ?? 'Unknown'}
                                {' · '}
                                {`${issue.comments?.count ?? 0} comments`}
                            </span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
