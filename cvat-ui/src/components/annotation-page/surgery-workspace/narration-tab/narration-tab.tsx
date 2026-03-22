// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import Divider from 'antd/lib/divider';

import NarrationRecorder from 'components/annotation-page/narration-recorder/narration-recorder';
import TranscriptViewer from '../transcript-viewer/transcript-viewer';

import './styles.scss';

export default function NarrationTab(): JSX.Element {
    return (
        <div className='cvat-narration-tab'>
            <NarrationRecorder />
            <Divider className='cvat-narration-tab-divider' />
            <TranscriptViewer />
        </div>
    );
}
