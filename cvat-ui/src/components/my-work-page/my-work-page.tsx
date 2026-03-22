// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useEffect, useCallback } from 'react';
import { useSelector } from 'react-redux';
import { useHistory } from 'react-router';
import Button from 'antd/lib/button';
import Empty from 'antd/lib/empty';
import Spin from 'antd/lib/spin';
import Tag from 'antd/lib/tag';
import notification from 'antd/lib/notification';
import { PlayCircleOutlined } from '@ant-design/icons';

import { CombinedState } from 'reducers';
import serverProxy from 'cvat-core/src/server-proxy';
import { SerializedJob } from 'cvat-core/src/server-response-types';
import { JobStage, JobState } from 'cvat-core/src/enums';

import './styles.scss';

const stageColors: Record<string, string> = {
    [JobStage.ANNOTATION]: 'blue',
    [JobStage.VALIDATION]: 'orange',
    [JobStage.ACCEPTANCE]: 'green',
};

const stateLabels: Record<string, string> = {
    [JobState.NEW]: 'New',
    [JobState.IN_PROGRESS]: 'In progress',
    [JobState.COMPLETED]: 'Completed',
    [JobState.REJECTED]: 'Rejected',
};

export default function MyWorkPage(): JSX.Element {
    const user = useSelector((state: CombinedState) => state.auth.user);
    const history = useHistory();

    const [jobs, setJobs] = useState<SerializedJob[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (!user) return undefined;
        let cancelled = false;
        setLoading(true);

        const filter = JSON.stringify({
            and: [
                { '==': [{ var: 'assignee' }, user.username] },
                { '!=': [{ var: 'state' }, JobState.COMPLETED] },
            ],
        });

        serverProxy.jobs
            .get({ filter, sort: '-updated_date', page_size: 50 })
            .then((data) => {
                if (!cancelled) setJobs(data);
            })
            .catch((err: unknown) => {
                if (!cancelled) {
                    notification.error({
                        message: 'Failed to load your work',
                        description: String(err),
                    });
                }
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });

        return () => { cancelled = true; };
    }, [user?.username]);

    const openJob = useCallback((job: SerializedJob) => {
        history.push(`/tasks/${job.task_id}/jobs/${job.id}`);
    }, [history]);

    if (!user) {
        return <Spin size='large' className='cvat-my-work-spinner' />;
    }

    return (
        <div className='cvat-my-work-page'>
            <div className='cvat-my-work-header'>
                <h2>My Work</h2>
                <span className='cvat-my-work-subtitle'>
                    Jobs assigned to
                    {' '}
                    {user.username}
                </span>
            </div>

            {loading ? (
                <div className='cvat-my-work-spinner'>
                    <Spin size='large' />
                </div>
            ) : jobs.length === 0 ? (
                <Empty
                    className='cvat-my-work-empty'
                    description='No jobs assigned to you right now.'
                />
            ) : (
                <div className='cvat-my-work-list'>
                    {jobs.map((job) => (
                        <div key={job.id} className='cvat-my-work-card'>
                            <div className='cvat-my-work-card-info'>
                                <span className='cvat-my-work-card-title'>
                                    {`Job #${job.id}`}
                                </span>
                                <span className='cvat-my-work-card-task'>
                                    {job.task_name}
                                    {job.project_name ? ` / ${job.project_name}` : ''}
                                </span>
                                <div className='cvat-my-work-card-tags'>
                                    <Tag color={stageColors[job.stage] ?? 'default'}>
                                        {job.stage}
                                    </Tag>
                                    <Tag>{stateLabels[job.state] ?? job.state}</Tag>
                                    <Tag>
                                        {`${job.frame_count} frames`}
                                    </Tag>
                                </div>
                            </div>
                            <Button
                                type='primary'
                                icon={<PlayCircleOutlined />}
                                className='cvat-my-work-card-open'
                                onClick={() => openJob(job)}
                            >
                                {job.state === JobState.NEW ? 'Start' : 'Resume'}
                            </Button>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
