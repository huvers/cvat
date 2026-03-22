// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useCallback, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useHistory } from 'react-router';
import Button from 'antd/lib/button';
import Layout from 'antd/lib/layout';
import Tabs from 'antd/lib/tabs';
import notification from 'antd/lib/notification';
import { CheckOutlined, DownloadOutlined } from '@ant-design/icons';

import { Job } from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import { finishCurrentJobAsync } from 'actions/annotation-actions';
import serverProxy from 'cvat-core/src/server-proxy';

import NarrationTab from '../narration-tab/narration-tab';
import PhaseTrackEditor from '../phase-track-editor/phase-track-editor';
import VideoClassificationEditor from '../video-classification-editor/video-classification-editor';
import './styles.scss';

export default function SurgerySidebar(): JSX.Element {
    const dispatch = useDispatch();
    const history = useHistory();
    const job = useSelector((state: CombinedState) => state.annotation.job.instance) as Job | null | undefined;
    const [submitting, setSubmitting] = useState(false);
    const [exporting, setExporting] = useState(false);

    const handleSubmitAndNext = useCallback(() => {
        if (!job) return;
        setSubmitting(true);
        dispatch(finishCurrentJobAsync(() => {
            setSubmitting(false);
            history.push('/my-work');
        }));
    }, [dispatch, history, job]);

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
                items={[
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
