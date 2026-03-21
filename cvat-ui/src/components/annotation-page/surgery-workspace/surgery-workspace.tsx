import './styles.scss';
import React from 'react';
import Layout from 'antd/lib/layout';

import CanvasLayout from 'components/annotation-page/canvas/grid-layout/canvas-layout';
import ControlsSideBarContainer from 'containers/annotation-page/standard-workspace/controls-side-bar/controls-side-bar';
import CanvasContextMenuContainer from 'containers/annotation-page/canvas/canvas-context-menu';
import CanvasPointContextMenuComponent from 'components/annotation-page/canvas/views/canvas2d/canvas-point-context-menu';
import RemoveConfirmComponent from 'components/annotation-page/standard-workspace/remove-confirm';
import BrushTools from 'components/annotation-page/canvas/views/canvas2d/brush-tools';

import SurgerySidebar from './surgery-sidebar/surgery-sidebar';

export default function SurgeryWorkspace(): JSX.Element {
    return (
        <Layout hasSider className='cvat-surgery-workspace'>
            <ControlsSideBarContainer />
            <CanvasLayout />
            <BrushTools />
            <SurgerySidebar />
            <CanvasContextMenuContainer />
            <CanvasPointContextMenuComponent />
            <RemoveConfirmComponent />
        </Layout>
    );
}
