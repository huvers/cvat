// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useEffect } from 'react';
import Button from 'antd/lib/button';
import Checkbox from 'antd/lib/checkbox';
import Input from 'antd/lib/input';
import InputNumber from 'antd/lib/input-number';
import Typography from 'antd/lib/typography';
import notification from 'antd/lib/notification';
import { CloudUploadOutlined } from '@ant-design/icons';

import serverProxy from 'cvat-core/src/server-proxy';

import './styles.scss';

export default function BulkIngestPage(): JSX.Element {
    const [cloudStorageId, setCloudStorageId] = useState<number | null>(null);
    const [procedurePrefix, setProcedurePrefix] = useState('');
    const [projectId, setProjectId] = useState<number | null>(null);
    const [triggerWeakLabeling, setTriggerWeakLabeling] = useState(false);
    const [submitting, setSubmitting] = useState(false);

    const handleSubmit = async (): Promise<void> => {
        if (!cloudStorageId || !procedurePrefix.trim()) return;
        setSubmitting(true);
        try {
            const { backendAPI } = (serverProxy as any).__internal?.config ?? {};
            const Axios = (await import('axios')).default;
            await Axios.post('/api/bulk-ingest', {
                cloud_storage_id: cloudStorageId,
                procedure_prefix: procedurePrefix.trim(),
                project_id: projectId,
                trigger_weak_labeling: triggerWeakLabeling,
            });
            notification.success({
                message: 'Bulk ingest started',
                description: `Scanning ${procedurePrefix} for episodes. Tasks will appear shortly.`,
            });
        } catch (err: unknown) {
            notification.error({
                message: 'Failed to start bulk ingest',
                description: String(err),
            });
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <div className='cvat-bulk-ingest-page'>
            <div className='cvat-bulk-ingest-card'>
                <Typography.Title level={3}>Bulk Ingest from S3</Typography.Title>
                <Typography.Paragraph type='secondary'>
                    Scan an S3 prefix for LeRobot-format surgical procedure videos and
                    auto-create CVAT tasks. Each episode becomes one task, auto-classified
                    with the procedure type.
                </Typography.Paragraph>

                <div className='cvat-bulk-ingest-form'>
                    <div className='cvat-bulk-ingest-field'>
                        <Typography.Text strong>Cloud Storage ID</Typography.Text>
                        <InputNumber
                            placeholder='e.g. 1'
                            value={cloudStorageId ?? undefined}
                            onChange={(v) => setCloudStorageId(v as number | null)}
                            style={{ width: '100%' }}
                        />
                        <Typography.Text type='secondary'>
                            ID of the S3 cloud storage registered in CVAT (Settings &gt; Cloud Storages)
                        </Typography.Text>
                    </div>

                    <div className='cvat-bulk-ingest-field'>
                        <Typography.Text strong>Procedure Prefix</Typography.Text>
                        <Input
                            placeholder='e.g. cholecystectomy'
                            value={procedurePrefix}
                            onChange={(e) => setProcedurePrefix(e.target.value)}
                        />
                        <Typography.Text type='secondary'>
                            {'S3 path prefix. Layout expected: {prefix}/videos/chunk-XXX/observation.images.endoscope/episode_XXXXXX.mp4'}
                        </Typography.Text>
                    </div>

                    <div className='cvat-bulk-ingest-field'>
                        <Typography.Text strong>Project ID (optional)</Typography.Text>
                        <InputNumber
                            placeholder='e.g. 5'
                            value={projectId ?? undefined}
                            onChange={(v) => setProjectId(v as number | null)}
                            style={{ width: '100%' }}
                        />
                        <Typography.Text type='secondary'>
                            If set, tasks will be created inside this project and use its labels.
                        </Typography.Text>
                    </div>

                    <Checkbox
                        checked={triggerWeakLabeling}
                        onChange={(e) => setTriggerWeakLabeling(e.target.checked)}
                    >
                        Trigger weak labeling after ingestion (if temporal models are available)
                    </Checkbox>

                    <Button
                        type='primary'
                        size='large'
                        icon={<CloudUploadOutlined />}
                        loading={submitting}
                        disabled={!cloudStorageId || !procedurePrefix.trim()}
                        onClick={handleSubmit}
                        block
                    >
                        Start Ingestion
                    </Button>
                </div>
            </div>
        </div>
    );
}
