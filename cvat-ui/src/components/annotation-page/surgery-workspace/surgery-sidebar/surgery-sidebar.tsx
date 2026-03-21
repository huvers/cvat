import React from 'react';
import Layout from 'antd/lib/layout';
import Tabs from 'antd/lib/tabs';
import Typography from 'antd/lib/typography';

import './styles.scss';

export default function SurgerySidebar(): JSX.Element {
    return (
        <Layout.Sider width={420} className='cvat-surgery-sidebar'>
            <Tabs
                className='cvat-surgery-sidebar-tabs'
                items={[
                    {
                        key: 'narration',
                        label: 'Narration',
                        children: (
                            <Typography.Paragraph className='cvat-surgery-sidebar-placeholder'>
                                Narration capture + transcript tooling will live here.
                            </Typography.Paragraph>
                        ),
                    },
                    {
                        key: 'phase-track',
                        label: 'Phase track',
                        children: (
                            <Typography.Paragraph className='cvat-surgery-sidebar-placeholder'>
                                Interval phase editor will live here.
                            </Typography.Paragraph>
                        ),
                    },
                ]}
            />
        </Layout.Sider>
    );
}
