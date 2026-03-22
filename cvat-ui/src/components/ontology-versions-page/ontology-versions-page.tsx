// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useCallback } from 'react';
import Button from 'antd/lib/button';
import Collapse from 'antd/lib/collapse';
import InputNumber from 'antd/lib/input-number';
import Spin from 'antd/lib/spin';
import Tag from 'antd/lib/tag';
import Typography from 'antd/lib/typography';
import notification from 'antd/lib/notification';
import { HistoryOutlined, SearchOutlined, DiffOutlined } from '@ant-design/icons';

import serverProxy from 'cvat-core/src/server-proxy';

import './styles.scss';

interface OntologyVersionData {
    id: number;
    version: number;
    schema: { labels: any[] };
    description: string;
    created_by_username: string | null;
    created_date: string;
}

interface DiffData {
    added: any[];
    removed: any[];
    renamed: { id: number; old_name: string; new_name: string }[];
    attributes_changed: any[];
}

export default function OntologyVersionsPage(): JSX.Element {
    const [projectId, setProjectId] = useState<number | null>(null);
    const [versions, setVersions] = useState<OntologyVersionData[]>([]);
    const [loading, setLoading] = useState(false);
    const [diff, setDiff] = useState<DiffData | null>(null);
    const [diffPair, setDiffPair] = useState<[number, number] | null>(null);

    const fetchVersions = useCallback(async () => {
        if (!projectId) return;
        setLoading(true);
        try {
            const data = await serverProxy.surgery.getOntologyVersions(projectId);
            setVersions(data as OntologyVersionData[]);
        } catch (err: unknown) {
            notification.error({ message: 'Failed to load versions', description: String(err) });
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    const fetchDiff = useCallback(async (v1Id: number, v2Id: number) => {
        try {
            const data = await serverProxy.surgery.getOntologyDiff(v1Id, v2Id);
            setDiff(data as DiffData);
            setDiffPair([v1Id, v2Id]);
        } catch (err: unknown) {
            notification.error({ message: 'Failed to load diff', description: String(err) });
        }
    }, []);

    const hasChanges = diff && (
        diff.added.length > 0 || diff.removed.length > 0
        || diff.renamed.length > 0 || diff.attributes_changed.length > 0
    );

    return (
        <div className='cvat-ontology-page'>
            <div className='cvat-ontology-header'>
                <Typography.Title level={3}>
                    <HistoryOutlined />
                    {' Ontology Version History'}
                </Typography.Title>
                <div className='cvat-ontology-controls'>
                    <InputNumber
                        placeholder='Project ID'
                        value={projectId ?? undefined}
                        onChange={(v) => setProjectId(v as number | null)}
                        style={{ width: 140 }}
                    />
                    <Button
                        type='primary'
                        icon={<SearchOutlined />}
                        onClick={fetchVersions}
                        disabled={!projectId}
                        loading={loading}
                    >
                        Load
                    </Button>
                </div>
            </div>

            {loading && versions.length === 0 && (
                <div className='cvat-ontology-spinner'><Spin size='large' /></div>
            )}

            {versions.length > 0 && (
                <div className='cvat-ontology-content'>
                    <div className='cvat-ontology-versions'>
                        <Typography.Text strong>
                            {`${versions.length} versions`}
                        </Typography.Text>
                        {versions.map((v, idx) => (
                            <div key={v.id} className='cvat-ontology-version-card'>
                                <div className='cvat-ontology-version-top'>
                                    <Tag color='blue'>{`v${v.version}`}</Tag>
                                    <span>{`${v.schema.labels.length} labels`}</span>
                                    <span className='cvat-ontology-version-date'>
                                        {new Date(v.created_date).toLocaleDateString()}
                                    </span>
                                </div>
                                <div className='cvat-ontology-version-meta'>
                                    {v.created_by_username ?? 'system'}
                                    {v.description ? ` — ${v.description}` : ''}
                                </div>
                                <div className='cvat-ontology-version-labels'>
                                    {v.schema.labels.map((l: any) => (
                                        <Tag key={l.id} color={l.color || undefined}>
                                            {l.name}
                                        </Tag>
                                    ))}
                                </div>
                                {idx < versions.length - 1 && (
                                    <Button
                                        size='small'
                                        icon={<DiffOutlined />}
                                        onClick={() => fetchDiff(versions[idx + 1].id, v.id)}
                                    >
                                        {`Diff vs v${versions[idx + 1].version}`}
                                    </Button>
                                )}
                            </div>
                        ))}
                    </div>

                    {diff && diffPair && (
                        <div className='cvat-ontology-diff'>
                            <Typography.Text strong>Diff</Typography.Text>
                            {!hasChanges ? (
                                <Typography.Paragraph type='secondary'>No changes.</Typography.Paragraph>
                            ) : (
                                <div className='cvat-ontology-diff-items'>
                                    {diff.added.map((l: any) => (
                                        <Tag key={`add-${l.id}`} color='success'>
                                            {`+ ${l.name}`}
                                        </Tag>
                                    ))}
                                    {diff.removed.map((l: any) => (
                                        <Tag key={`rm-${l.id}`} color='error'>
                                            {`- ${l.name}`}
                                        </Tag>
                                    ))}
                                    {diff.renamed.map((r) => (
                                        <Tag key={`ren-${r.id}`} color='warning'>
                                            {`${r.old_name} → ${r.new_name}`}
                                        </Tag>
                                    ))}
                                    {diff.attributes_changed.map((a: any) => (
                                        <Tag key={`attr-${a.label_id}`} color='processing'>
                                            {`${a.label}: attrs changed`}
                                        </Tag>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
