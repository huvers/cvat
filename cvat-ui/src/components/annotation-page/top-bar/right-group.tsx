// Copyright (C) 2020-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useEffect, useCallback, useState } from 'react';
import { Col } from 'antd/lib/grid';
import Icon, { InfoCircleOutlined } from '@ant-design/icons';
import Select from 'antd/lib/select';
import Button from 'antd/lib/button';
import Modal from 'antd/lib/modal';
import notification from 'antd/lib/notification';

import { FilterIcon, FullscreenIcon, GuideIcon } from 'icons';
import config from 'config';
import { isDev } from 'utils/environment';
import {
    DimensionType, Job, JobStage, JobState,
} from 'cvat-core-wrapper';
import { Workspace } from 'reducers';

import MDEditor from '@uiw/react-md-editor';
import NarrationRecorder from 'components/annotation-page/narration-recorder/narration-recorder';

interface Props {
    showStatistics(): void;
    showFilters(): void;
    changeWorkspace(workspace: Workspace): void;
    jobInstance: Job;
    workspace: Workspace;
    annotationFilters: object[];
    initialOpenGuide: boolean;
}

function RightGroup(props: Props): JSX.Element {
    const {
        showStatistics,
        changeWorkspace,
        showFilters,
        workspace,
        jobInstance,
        annotationFilters,
        initialOpenGuide,
    } = props;

    const filters = annotationFilters.length;
    const [narrationModalVisible, setNarrationModalVisible] = useState(false);
    const narrationRecorderEnabled = isDev() || (() => {
        try {
            return localStorage.getItem('enableNarrationRecorder') === 'true';
        } catch (error: unknown) {
            return false;
        }
    })();

    // Surgery workspace is always enabled in this fork
    const surgeryWorkspaceEnabled = true;

    const openGuide = useCallback(() => {
        const PADDING = Math.min(window.screen.availHeight, window.screen.availWidth) * 0.4;
        jobInstance.guide().then((guide) => {
            if (guide) {
                Modal.info({
                    icon: null,
                    width: window.screen.availWidth - PADDING,
                    className: 'cvat-annotation-view-markdown-guide-modal',
                    content: (
                        <MDEditor
                            visibleDragbar={false}
                            data-color-mode='light'
                            height={window.screen.availHeight - PADDING}
                            preview='preview'
                            hideToolbar
                            value={guide.markdown}
                        />
                    ),
                });
            }
        }).catch((error: unknown) => {
            notification.error({
                message: 'Could not receive annotation guide',
                description: error instanceof Error ? error.message : console.error('error'),
            });
        });
    }, [jobInstance]);

    useEffect(() => {
        if (Number.isInteger(jobInstance?.guideId)) {
            if (initialOpenGuide) {
                openGuide();
            } else if (
                jobInstance?.stage === JobStage.ANNOTATION &&
                jobInstance?.state === JobState.NEW
            ) {
                let seenGuides = [];
                try {
                    seenGuides = JSON.parse(localStorage.getItem('seenGuides') || '[]');
                    if (!Array.isArray(seenGuides) || seenGuides.some((el) => !Number.isInteger(el))) {
                        throw new Error('Wrong structure stored in local storage');
                    }
                } catch (error: unknown) {
                    seenGuides = [];
                }

                if (!seenGuides.includes(jobInstance.guideId)) {
                    // open guide if the user have not seen it yet
                    openGuide();
                    const updatedSeenGuides = Array
                        .from(new Set([
                            jobInstance.guideId,
                            ...seenGuides.slice(0, config.LOCAL_STORAGE_SEEN_GUIDES_MEMORY_LIMIT - 1),
                        ]));
                    localStorage.setItem('seenGuides', JSON.stringify(updatedSeenGuides));
                }
            }
        }
    }, []);

    return (
        <Col className='cvat-annotation-header-right-group'>
            <Button
                type='link'
                className='cvat-annotation-header-fullscreen-button cvat-annotation-header-button'
                onClick={(): void => {
                    if (window.document.fullscreenEnabled) {
                        if (window.document.fullscreenElement) {
                            window.document.exitFullscreen();
                        } else {
                            window.document.documentElement.requestFullscreen();
                        }
                    }
                }}
            >
                <Icon component={FullscreenIcon} />
                Fullscreen
            </Button>
            { jobInstance.guideId !== null && (
                <Button
                    type='link'
                    className='cvat-annotation-header-guide-button cvat-annotation-header-button'
                    onClick={openGuide}
                >
                    <Icon component={GuideIcon} />
                    Guide
                </Button>
            )}
            <Button
                type='link'
                className='cvat-annotation-header-info-button cvat-annotation-header-button'
                onClick={showStatistics}
            >
                <InfoCircleOutlined />
                Info
            </Button>
            {narrationRecorderEnabled && (
                <Button
                    type='link'
                    className='cvat-annotation-header-narration-button cvat-annotation-header-button'
                    onClick={() => setNarrationModalVisible(true)}
                >
                    Narration
                </Button>
            )}
            <Button
                type='link'
                className={`cvat-annotation-header-filters-button cvat-annotation-header-button ${filters ?
                    'filters-armed' : ''
                }`}
                onClick={showFilters}
            >
                <Icon component={FilterIcon} />
                Filters
            </Button>
            <div>
                <Select
                    popupClassName='cvat-workspace-selector-dropdown'
                    className='cvat-workspace-selector'
                    onChange={changeWorkspace}
                    value={workspace}
                >
                    {Object.values(Workspace).map((ws) => {
                        if (ws === Workspace.SURGERY && !surgeryWorkspaceEnabled) {
                            return null;
                        }

                        if (jobInstance.dimension === DimensionType.DIMENSION_3D) {
                            if (ws === Workspace.STANDARD) {
                                return null;
                            }
                            if (ws === Workspace.SURGERY) {
                                return null;
                            }
                            return (
                                <Select.Option disabled={ws !== Workspace.STANDARD3D} key={ws} value={ws}>
                                    {ws}
                                </Select.Option>
                            );
                        }
                        if (ws !== Workspace.STANDARD3D) {
                            return (
                                <Select.Option key={ws} value={ws}>
                                    {ws}
                                </Select.Option>
                            );
                        }
                        return null;
                    })}
                </Select>
            </div>
            <Modal
                title='Narration'
                open={narrationModalVisible}
                onCancel={() => setNarrationModalVisible(false)}
                footer={null}
                width={520}
                destroyOnClose
            >
                <NarrationRecorder />
            </Modal>
        </Col>
    );
}

export default React.memo(RightGroup);
