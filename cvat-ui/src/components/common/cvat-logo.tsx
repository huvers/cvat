// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import { useSelector } from 'react-redux';
import { CombinedState } from 'reducers';
import { PROJECT_NAME } from 'branding';

interface Props {
    className?: string;
}

function CVATLogo({ className = '' }: Props): JSX.Element {
    const logo = useSelector((state: CombinedState) => state.about.server.logoURL);

    return (
        <div className={`cvat-logo-icon ${className}`.trim()}>
            <img src={logo} alt={`${PROJECT_NAME} logo`} />
        </div>
    );
}

export default React.memo(CVATLogo);
