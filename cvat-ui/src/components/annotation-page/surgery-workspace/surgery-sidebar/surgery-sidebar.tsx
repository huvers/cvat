import React from 'react';
import Layout from 'antd/lib/layout';
import Tabs from 'antd/lib/tabs';

import PhaseTrackEditor from '../phase-track-editor/phase-track-editor';
import TranscriptViewer from '../transcript-viewer/transcript-viewer';
import VideoClassificationEditor from '../video-classification-editor/video-classification-editor';
import './styles.scss';

export default function SurgerySidebar(): JSX.Element {
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
                        children: <TranscriptViewer />,
                    },
                    {
                        key: 'phase-track',
                        label: 'Phase track',
                        children: <PhaseTrackEditor />,
                    },
                ]}
            />
        </Layout.Sider>
    );
}
