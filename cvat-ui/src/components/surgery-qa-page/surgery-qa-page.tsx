// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useCallback } from 'react';
import Button from 'antd/lib/button';
import InputNumber from 'antd/lib/input-number';
import Progress from 'antd/lib/progress';
import Spin from 'antd/lib/spin';
import Table from 'antd/lib/table';
import Tag from 'antd/lib/tag';
import Typography from 'antd/lib/typography';
import notification from 'antd/lib/notification';
import { SearchOutlined } from '@ant-design/icons';

import './styles.scss';

interface JobMetrics {
    job_id: number;
    task_name: string;
    stage: string;
    state: string;
    assignee: string | null;
    total_frames: number;
    coverage: { coverage_pct: number; gap_count: number };
    intervals: { total: number; manual: number; auto: number; density_per_1k: number };
    issues: { total: number; open: number; resolved: number };
    narrations: number;
    transcripts: Record<string, number>;
    classifications: string[];
    completeness: Record<string, boolean>;
}

interface ProjectMetrics {
    project_id: number;
    summary: {
        total_jobs: number;
        completed_jobs: number;
        completion_pct: number;
        avg_coverage_pct: number;
        total_open_issues: number;
        total_intervals: number;
        total_auto_intervals: number;
        fully_complete_jobs: number;
    };
    jobs: JobMetrics[];
}

const stateColors: Record<string, string> = {
    new: 'default',
    'in progress': 'processing',
    completed: 'success',
    rejected: 'error',
};

const columns = [
    {
        title: 'Job',
        dataIndex: 'job_id',
        key: 'job_id',
        width: 70,
        render: (id: number) => `#${id}`,
    },
    {
        title: 'Task',
        dataIndex: 'task_name',
        key: 'task_name',
        ellipsis: true,
    },
    {
        title: 'Assignee',
        dataIndex: 'assignee',
        key: 'assignee',
        width: 100,
        render: (v: string | null) => v ?? '—',
    },
    {
        title: 'State',
        dataIndex: 'state',
        key: 'state',
        width: 100,
        render: (v: string) => <Tag color={stateColors[v] ?? 'default'}>{v}</Tag>,
    },
    {
        title: 'Coverage',
        key: 'coverage',
        width: 110,
        render: (_: unknown, row: JobMetrics) => (
            <Progress
                percent={row.coverage.coverage_pct}
                size='small'
                status={row.coverage.coverage_pct >= 90 ? 'success' : 'normal'}
            />
        ),
    },
    {
        title: 'Intervals',
        key: 'intervals',
        width: 100,
        render: (_: unknown, row: JobMetrics) => (
            <span>
                {row.intervals.total}
                {row.intervals.auto > 0 && (
                    <Tag color='blue' style={{ marginLeft: 4, fontSize: 10 }}>
                        {`${row.intervals.auto} auto`}
                    </Tag>
                )}
            </span>
        ),
    },
    {
        title: 'Issues',
        key: 'issues',
        width: 80,
        render: (_: unknown, row: JobMetrics) => (
            row.issues.open > 0
                ? <Tag color='warning'>{`${row.issues.open} open`}</Tag>
                : <span>{row.issues.total}</span>
        ),
    },
    {
        title: 'Complete',
        key: 'completeness',
        width: 90,
        render: (_: unknown, row: JobMetrics) => {
            const checks = Object.values(row.completeness);
            const done = checks.filter(Boolean).length;
            return (
                <span>
                    {`${done}/${checks.length}`}
                    {done === checks.length && ' ✓'}
                </span>
            );
        },
    },
];

export default function SurgeryQAPage(): JSX.Element {
    const [projectId, setProjectId] = useState<number | null>(null);
    const [loading, setLoading] = useState(false);
    const [metrics, setMetrics] = useState<ProjectMetrics | null>(null);

    const fetchMetrics = useCallback(async () => {
        if (!projectId) return;
        setLoading(true);
        try {
            const axios = (await import('axios')).default;
            const response = await axios.get('/api/surgery-qa', {
                params: { project_id: projectId },
            });
            setMetrics(response.data);
        } catch (err: unknown) {
            notification.error({ message: 'Failed to load QA metrics', description: String(err) });
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    const summary = metrics?.summary;

    return (
        <div className='cvat-surgery-qa-page'>
            <div className='cvat-surgery-qa-header'>
                <Typography.Title level={3}>Surgery QA Dashboard</Typography.Title>
                <div className='cvat-surgery-qa-controls'>
                    <InputNumber
                        placeholder='Project ID'
                        value={projectId ?? undefined}
                        onChange={(v) => setProjectId(v as number | null)}
                        style={{ width: 140 }}
                    />
                    <Button
                        type='primary'
                        icon={<SearchOutlined />}
                        onClick={fetchMetrics}
                        disabled={!projectId}
                        loading={loading}
                    >
                        Load
                    </Button>
                </div>
            </div>

            {loading && !metrics && (
                <div className='cvat-surgery-qa-spinner'>
                    <Spin size='large' />
                </div>
            )}

            {summary && (
                <>
                    <div className='cvat-surgery-qa-summary'>
                        <div className='cvat-surgery-qa-stat'>
                            <span className='cvat-surgery-qa-stat-value'>{summary.total_jobs}</span>
                            <span className='cvat-surgery-qa-stat-label'>Total Jobs</span>
                        </div>
                        <div className='cvat-surgery-qa-stat'>
                            <span className='cvat-surgery-qa-stat-value'>
                                {`${summary.completion_pct}%`}
                            </span>
                            <span className='cvat-surgery-qa-stat-label'>Completed</span>
                        </div>
                        <div className='cvat-surgery-qa-stat'>
                            <span className='cvat-surgery-qa-stat-value'>
                                {`${summary.avg_coverage_pct}%`}
                            </span>
                            <span className='cvat-surgery-qa-stat-label'>Avg Coverage</span>
                        </div>
                        <div className='cvat-surgery-qa-stat'>
                            <span className='cvat-surgery-qa-stat-value'>{summary.total_open_issues}</span>
                            <span className='cvat-surgery-qa-stat-label'>Open Issues</span>
                        </div>
                        <div className='cvat-surgery-qa-stat'>
                            <span className='cvat-surgery-qa-stat-value'>{summary.total_intervals}</span>
                            <span className='cvat-surgery-qa-stat-label'>
                                {`Intervals (${summary.total_auto_intervals} auto)`}
                            </span>
                        </div>
                        <div className='cvat-surgery-qa-stat'>
                            <span className='cvat-surgery-qa-stat-value'>{summary.fully_complete_jobs}</span>
                            <span className='cvat-surgery-qa-stat-label'>Fully Complete</span>
                        </div>
                    </div>

                    <Table
                        className='cvat-surgery-qa-table'
                        dataSource={metrics.jobs}
                        columns={columns}
                        rowKey='job_id'
                        size='small'
                        pagination={{ pageSize: 20 }}
                    />
                </>
            )}
        </div>
    );
}
