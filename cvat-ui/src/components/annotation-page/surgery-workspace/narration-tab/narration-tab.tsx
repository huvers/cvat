// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useCallback, useState } from 'react';
import Divider from 'antd/lib/divider';

import NarrationRecorder from 'components/annotation-page/narration-recorder/narration-recorder';
import TranscriptViewer from '../transcript-viewer/transcript-viewer';

import './styles.scss';

export default function NarrationTab(): JSX.Element {
    const [refreshKey, setRefreshKey] = useState(0);

    const handleUploadSuccess = useCallback(() => {
        setRefreshKey((k) => k + 1);
    }, []);

    return (
        <div className='cvat-narration-tab'>
            <NarrationRecorder onUploadSuccess={handleUploadSuccess} />
            <Divider className='cvat-narration-tab-divider' />
            <TranscriptViewer refreshKey={refreshKey} />
        </div>
    );
}
