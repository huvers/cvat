// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useEffect, useCallback } from 'react';
import Button from 'antd/lib/button';
import Progress from 'antd/lib/progress';
import Spin from 'antd/lib/spin';
import Table from 'antd/lib/table';
import Tag from 'antd/lib/tag';
import Typography from 'antd/lib/typography';
import notification from 'antd/lib/notification';
import {
    SyncOutlined, CloudUploadOutlined, DatabaseOutlined,
    DownloadOutlined,
} from '@ant-design/icons';

import serverProxy from 'cvat-core/src/server-proxy';

import './styles.scss';

interface DatasetData {
    id: number;
    name: string;
    procedure_type: string;
    cloud_storage_id: number;
    s3_prefix: string;
    project_id: number | null;
    episode_counts: Record<string, number>;
    created_date: string;
}

const STATUS_COLORS: Record<string, string> = {
    discovered: 'default',
    ingested: 'processing',
    annotated: 'success',
    exported: 'purple',
};

export default function DatasetsPage(): JSX.Element {
    const [datasets, setDatasets] = useState<DatasetData[]>([]);
    const [loading, setLoading] = useState(false);
    const [syncing, setSyncing] = useState<number | null>(null);
    const [ingesting, setIngesting] = useState<number | null>(null);

    const fetchDatasets = useCallback(async () => {
        setLoading(true);
        try {
            const data = await serverProxy.surgery.getDatasets();
            setDatasets(data as DatasetData[]);
        } catch (err: unknown) {
            notification.error({ message: 'Failed to load datasets', description: String(err) });
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchDatasets();
    }, [fetchDatasets]);

    const handleSync = useCallback(async (id: number) => {
        setSyncing(id);
        try {
            const result = await serverProxy.surgery.syncDataset(id);
            notification.success({
                message: 'Sync complete',
                description: `${(result as any).new} new episodes discovered, ${(result as any).existing} existing.`,
            });
            fetchDatasets();
        } catch (err: unknown) {
            notification.error({ message: 'Sync failed', description: String(err) });
        } finally {
            setSyncing(null);
        }
    }, [fetchDatasets]);

    const handleIngest = useCallback(async (id: number) => {
        setIngesting(id);
        try {
            await serverProxy.surgery.ingestDataset(id);
            notification.success({ message: 'Ingest queued', description: 'Tasks will be created shortly.' });
        } catch (err: unknown) {
            notification.error({ message: 'Ingest failed', description: String(err) });
        } finally {
            setIngesting(null);
        }
    }, []);

    const handleExport = useCallback(async (id: number, fmt: 'coco' | 'temporal') => {
        try {
            const data = fmt === 'coco'
                ? await serverProxy.surgery.exportDatasetCoco(id)
                : await serverProxy.surgery.exportDatasetTemporal(id);
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `dataset_${id}_${fmt}.json`;
            a.click();
            URL.revokeObjectURL(url);
            notification.success({ message: `${fmt.toUpperCase()} export downloaded` });
        } catch (err: unknown) {
            notification.error({ message: 'Export failed', description: String(err) });
        }
    }, []);

    const columns = [
        {
            title: 'Name',
            dataIndex: 'name',
            key: 'name',
            render: (name: string, row: DatasetData) => (
                <span>
                    <DatabaseOutlined style={{ marginRight: 6 }} />
                    <strong>{name}</strong>
                </span>
            ),
        },
        {
            title: 'Procedure',
            dataIndex: 'procedure_type',
            key: 'procedure_type',
            render: (v: string) => <Tag>{v}</Tag>,
        },
        {
            title: 'Episodes',
            key: 'episodes',
            render: (_: unknown, row: DatasetData) => {
                const c = row.episode_counts;
                const total = c.total || 0;
                const ingested = (c.ingested || 0) + (c.annotated || 0) + (c.exported || 0);
                const pct = total > 0 ? Math.round((ingested / total) * 100) : 0;
                return (
                    <div className='cvat-dataset-episodes-col'>
                        <Progress percent={pct} size='small' style={{ width: 80 }} />
                        <span className='cvat-dataset-episode-counts'>
                            {Object.entries(c).filter(([k]) => k !== 'total').map(([status, count]) => (
                                <Tag key={status} color={STATUS_COLORS[status] ?? 'default'}>
                                    {`${count} ${status}`}
                                </Tag>
                            ))}
                        </span>
                    </div>
                );
            },
        },
        {
            title: 'Actions',
            key: 'actions',
            width: 320,
            render: (_: unknown, row: DatasetData) => (
                <span className='cvat-dataset-actions'>
                    <Button
                        size='small'
                        icon={<SyncOutlined spin={syncing === row.id} />}
                        loading={syncing === row.id}
                        onClick={() => handleSync(row.id)}
                    >
                        Sync
                    </Button>
                    <Button
                        size='small'
                        type='primary'
                        icon={<CloudUploadOutlined />}
                        loading={ingesting === row.id}
                        onClick={() => handleIngest(row.id)}
                    >
                        Ingest
                    </Button>
                    <Button
                        size='small'
                        icon={<DownloadOutlined />}
                        onClick={() => handleExport(row.id, 'coco')}
                    >
                        COCO
                    </Button>
                    <Button
                        size='small'
                        icon={<DownloadOutlined />}
                        onClick={() => handleExport(row.id, 'temporal')}
                    >
                        Temporal
                    </Button>
                </span>
            ),
        },
    ];

    return (
        <div className='cvat-datasets-page'>
            <div className='cvat-datasets-header'>
                <Typography.Title level={3}>
                    <DatabaseOutlined />
                    {' Datasets'}
                </Typography.Title>
            </div>

            {loading && datasets.length === 0 ? (
                <div className='cvat-datasets-spinner'><Spin size='large' /></div>
            ) : (
                <Table
                    dataSource={datasets}
                    columns={columns}
                    rowKey='id'
                    pagination={false}
                />
            )}
        </div>
    );
}
