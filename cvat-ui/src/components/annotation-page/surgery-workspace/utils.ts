// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

/**
 * Surgical video standard: 50fps recorded, upsampled to 60fps for kinematics.
 */
export const SURGERY_FPS = 60;

/**
 * Convert a frame number to a mm:ss time string.
 */
export function frameToTime(frame: number, startFrame: number = 0, fps: number = SURGERY_FPS): string {
    const seconds = (frame - startFrame) / fps;
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}
