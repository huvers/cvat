// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useCallback, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useHistory } from 'react-router';
import Button from 'antd/lib/button';
import Layout from 'antd/lib/layout';
import Modal from 'antd/lib/modal';
import Tabs from 'antd/lib/tabs';
import notification from 'antd/lib/notification';
import { CheckOutlined, DownloadOutlined } from '@ant-design/icons';

import { Job } from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import { finishCurrentJobAsync } from 'actions/annotation-actions';
import serverProxy from 'cvat-core/src/server-proxy';

import CopilotPanel from '../copilot-panel/copilot-panel';
import NarrationTab from '../narration-tab/narration-tab';
import PhaseTrackEditor from '../phase-track-editor/phase-track-editor';
import SurgeryIssues from '../surgery-issues/surgery-issues';
import VideoClassificationEditor from '../video-classification-editor/video-classification-editor';
import './styles.scss';

export default function SurgerySidebar(): JSX.Element {
    const dispatch = useDispatch();
    const history = useHistory();
    const job = useSelector((state: CombinedState) => state.annotation.job.instance) as Job | null | undefined;
    const [submitting, setSubmitting] = useState(false);
    const [exporting, setExporting] = useState(false);

    const doSubmit = useCallback(async () => {
        if (!job) return;
        setSubmitting(true);
        try {
            await dispatch(finishCurrentJobAsync(() => {
                history.push('/my-work');
            }));
        } catch (err: unknown) {
            notification.error({
                message: 'Failed to submit job',
                description: err instanceof Error ? err.message : String(err),
            });
        } finally {
            setSubmitting(false);
        }
    }, [dispatch, history, job]);

    const handleSubmitAndNext = useCallback(async () => {
        if (!job) return;
        // Fetch metrics for the confirmation summary
        try {
            const metrics = await serverProxy.jobs.getSurgeryMetrics(job.id);
            const cov = metrics.coverage?.coverage_pct ?? '?';
            const intervals = metrics.intervals?.total ?? '?';
            const autoCount = metrics.intervals?.auto ?? 0;
            const openIssues = metrics.issues?.open ?? 0;
            const narrations = metrics.narrations ?? 0;

            Modal.confirm({
                title: 'Submit this job?',
                content: (
                    <div style={{ lineHeight: 1.8 }}>
                        <div>{`Coverage: ${cov}%`}</div>
                        <div>{`Phases: ${intervals}${autoCount > 0 ? ` (${autoCount} auto)` : ''}`}</div>
                        <div>{`Narrations: ${narrations}`}</div>
                        {openIssues > 0 && (
                            <div style={{ color: '#faad14' }}>{`${openIssues} open issue${openIssues > 1 ? 's' : ''}`}</div>
                        )}
                    </div>
                ),
                okText: 'Submit & Next',
                cancelText: 'Cancel',
                onOk: doSubmit,
            });
        } catch {
            // If metrics fail, submit anyway with basic confirm
            Modal.confirm({
                title: 'Submit this job and move to the next?',
                okText: 'Submit & Next',
                cancelText: 'Cancel',
                onOk: doSubmit,
            });
        }
    }, [job, doSubmit]);

    const handleExport = useCallback(async () => {
        if (!job) return;
        setExporting(true);
        try {
            const data = await serverProxy.jobs.getSurgeryExport(job.id);
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `surgery_export_job_${job.id}.json`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (err: unknown) {
            notification.error({ message: 'Export failed', description: String(err) });
        } finally {
            setExporting(false);
        }
    }, [job]);

    return (
        <Layout.Sider width={420} className='cvat-surgery-sidebar'>
            <Tabs
                className='cvat-surgery-sidebar-tabs'
                defaultActiveKey='copilot'
                items={[
                    {
                        key: 'copilot',
                        label: 'Copilot',
                        children: <CopilotPanel />,
                    },
                    {
                        key: 'case-info',
                        label: 'Case info',
                        children: <VideoClassificationEditor />,
                    },
                    {
                        key: 'narration',
                        label: 'Narration',
                        children: <NarrationTab />,
                    },
                    {
                        key: 'phase-track',
                        label: 'Phase track',
                        children: <PhaseTrackEditor />,
                    },
                    {
                        key: 'issues',
                        label: 'Issues',
                        children: <SurgeryIssues />,
                    },
                ]}
            />
            <div className='cvat-surgery-sidebar-footer'>
                <Button
                    icon={<DownloadOutlined />}
                    onClick={handleExport}
                    loading={exporting}
                    disabled={!job}
                >
                    Export JSON
                </Button>
                <Button
                    type='primary'
                    icon={<CheckOutlined />}
                    onClick={handleSubmitAndNext}
                    loading={submitting}
                    disabled={!job}
                >
                    Submit & Next
                </Button>
            </div>
        </Layout.Sider>
    );
}
