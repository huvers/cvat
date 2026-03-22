// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useCallback } from 'react';
import { useSelector } from 'react-redux';
import { useHistory } from 'react-router';
import Button from 'antd/lib/button';
import Input from 'antd/lib/input';
import Select from 'antd/lib/select';
import Typography from 'antd/lib/typography';
import notification from 'antd/lib/notification';
import { UserOutlined } from '@ant-design/icons';

import { CombinedState } from 'reducers';
import serverProxy from 'cvat-core/src/server-proxy';

import './styles.scss';

const ROLES = [
    { value: 'surgeon', label: 'Surgeon' },
    { value: 'engineer', label: 'Engineer' },
    { value: 'researcher', label: 'Researcher' },
    { value: 'reviewer', label: 'Reviewer' },
];

const EXPERTISE_LEVELS = [
    { value: 'resident', label: 'Resident' },
    { value: 'fellow', label: 'Fellow' },
    { value: 'attending', label: 'Attending' },
    { value: 'expert', label: 'Expert' },
];

export default function ProfileSetup(): JSX.Element {
    const user = useSelector((state: CombinedState) => state.auth.user);
    const history = useHistory();

    const [role, setRole] = useState('');
    const [expertiseLevel, setExpertiseLevel] = useState('');
    const [specialty, setSpecialty] = useState('');
    const [institution, setInstitution] = useState('');
    const [saving, setSaving] = useState(false);

    const canSave = role && expertiseLevel;

    const handleSave = useCallback(async () => {
        if (!user || !canSave) return;
        setSaving(true);
        try {
            await serverProxy.users.update(user.id, {
                role,
                expertise_level: expertiseLevel,
                specialty,
                institution,
            });
            notification.success({ message: 'Profile saved!' });
            history.push('/my-work');
        } catch (err: unknown) {
            notification.error({ message: 'Failed to save profile', description: String(err) });
        } finally {
            setSaving(false);
        }
    }, [user, role, expertiseLevel, specialty, institution, canSave, history]);

    return (
        <div className='cvat-profile-setup-page'>
            <div className='cvat-profile-setup-card'>
                <UserOutlined className='cvat-profile-setup-icon' />
                <Typography.Title level={3}>Complete Your Profile</Typography.Title>
                <Typography.Paragraph type='secondary'>
                    Tell us about your background so we can tailor the annotation experience
                    and track expertise levels for data quality.
                </Typography.Paragraph>

                <div className='cvat-profile-setup-form'>
                    <div className='cvat-profile-setup-field'>
                        <Typography.Text strong>Role *</Typography.Text>
                        <Select
                            placeholder='Select your role'
                            value={role || undefined}
                            onChange={(v: string) => setRole(v)}
                            options={ROLES}
                            style={{ width: '100%' }}
                        />
                    </div>

                    <div className='cvat-profile-setup-field'>
                        <Typography.Text strong>Expertise Level *</Typography.Text>
                        <Select
                            placeholder='Select your expertise level'
                            value={expertiseLevel || undefined}
                            onChange={(v: string) => setExpertiseLevel(v)}
                            options={EXPERTISE_LEVELS}
                            style={{ width: '100%' }}
                        />
                    </div>

                    <div className='cvat-profile-setup-field'>
                        <Typography.Text strong>Specialty</Typography.Text>
                        <Input
                            placeholder='e.g., General Surgery, Urology'
                            value={specialty}
                            onChange={(e) => setSpecialty(e.target.value)}
                        />
                    </div>

                    <div className='cvat-profile-setup-field'>
                        <Typography.Text strong>Institution</Typography.Text>
                        <Input
                            placeholder='e.g., Johns Hopkins, Mayo Clinic'
                            value={institution}
                            onChange={(e) => setInstitution(e.target.value)}
                        />
                    </div>

                    <Button
                        type='primary'
                        size='large'
                        block
                        disabled={!canSave}
                        loading={saving}
                        onClick={handleSave}
                    >
                        Save & Continue
                    </Button>
                </div>
            </div>
        </div>
    );
}
