// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, {
    useCallback, useEffect, useMemo, useRef, useState,
} from 'react';
import { useDispatch, useSelector } from 'react-redux';
import Alert from 'antd/lib/alert';
import Button from 'antd/lib/button';
import Divider from 'antd/lib/divider';
import Empty from 'antd/lib/empty';
import notification from 'antd/lib/notification';
import Select from 'antd/lib/select';
import Slider from 'antd/lib/slider';
import Space from 'antd/lib/space';
import Spin from 'antd/lib/spin';
import Typography from 'antd/lib/typography';
import {
    CheckOutlined, FastForwardOutlined, PlusOutlined, ReloadOutlined, ScissorOutlined,
    StopOutlined,
} from '@ant-design/icons';
import Collapse from 'antd/lib/collapse';
import InputNumber from 'antd/lib/input-number';
import Progress from 'antd/lib/progress';

import {
    getCore, Job, Label, ObjectState, ObjectType, ShapeType,
} from 'cvat-core-wrapper';
import { Source } from 'cvat-core/src/enums';
import {
    SerializedSAM3MaskResult,
    SerializedSAM3Model,
    SerializedSAM3PointsResult,
    SerializedSAM3PromptOption,
    SerializedSAM3PropagationResult,
    SerializedSAM3TextResult,
    SerializedSAM3VideoMask,
    SerializedSAM3VideoSession,
} from 'cvat-core/src/server-response-types';
import serverProxy from 'cvat-core/src/server-proxy';
import { Canvas, convertShapesForInteractor, InteractionResult } from 'cvat-canvas-wrapper';
import { activateObject, fetchAnnotationsAsync } from 'actions/annotation-actions';
import { CombinedState } from 'reducers';

type SAM3PanelModel = SerializedSAM3Model;
type SAM3PromptOption = SerializedSAM3PromptOption;
type PointMode = 0 | 1;

interface SessionState {
    sessionID: string;
    frame: number;
    modelID: number;
    width: number;
    height: number;
    confidenceThreshold: number;
}

interface PointSessionState {
    targetClientID: number | null;
    sourceClientID: number | null;
    auxiliaryClientIDs: number[];
    createdDuringSession: boolean;
    replacedExistingObject: boolean;
    originalState: ShapeSnapshot | null;
    logitsToken: string | null;
    mode: PointMode;
}

interface VideoTrackingState {
    sessionID: string;
    modelID: number;
    startFrame: number;
    stopFrame: number;
    numFrames: number;
    width: number;
    height: number;
    hasPrompt: boolean;
    objectLabelIDs: Record<number, number>;
    seededSourceClientIDsByFrame: Record<number, number[]>;
    seededObjectIDsByFrame: Record<number, Record<number, number>>;
    generatedClientIDsByFrame: Record<number, number[]>;
    generatedObjectIDsByFrame: Record<number, Record<number, number>>;
    propagating: boolean;
    progress: number;
    results: SerializedSAM3PropagationResult[];
}

interface PromptBinding {
    option: SAM3PromptOption;
    label: Label | null;
}

interface AutoMaskResult {
    label: Label;
    maskRle: number[] | null;
    confidence: number;
    autoMaskPriority: number;
}

interface ShapeSnapshot {
    label: Label;
    frame: number;
    zOrder: number;
    shapeType: ShapeType;
    source: Source;
    points: number[];
    occluded: boolean;
    attributes: Record<number, string>;
    descriptions: string[];
    rotation?: number | null;
}

interface PromptTargetOptions {
    preferSelectedPrompt?: boolean;
}

interface MaterializedMaskCandidate {
    rle: number[];
    fullMask: Uint8Array;
    area: number;
}

interface MaskMutationResult {
    primaryClientID: number;
    auxiliaryClientIDs: number[];
}

interface VideoAnnotationApplyResult {
    generatedClientIDsByFrame: Record<number, number[]>;
    generatedObjectIDsByFrame: Record<number, Record<number, number>>;
}

const core = getCore();
const { Text } = Typography;
const DEFAULT_UNION_MIN_COMPONENT_AREA = 1024;
const DEFAULT_SPLIT_MIN_COMPONENT_AREA = 192;

function isEligibleMaskLabel(label: Label): boolean {
    return ['mask', 'any'].includes(label.type);
}

function normalizePromptToken(value: string): string {
    return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function normalizeBBox(points: number[]): number[] {
    return [
        Math.min(points[0], points[2]),
        Math.min(points[1], points[3]),
        Math.max(points[0], points[2]),
        Math.max(points[1], points[3]),
    ];
}

function encodeBBoxToNormalizedXYWH(bbox: number[] | null, width: number, height: number): number[] | null {
    if (!bbox || bbox.length !== 4 || width <= 0 || height <= 0) {
        return null;
    }

    const [x0, y0, x1, y1] = normalizeBBox(bbox);
    const clampedX0 = Math.max(0, Math.min(width - 1, x0));
    const clampedY0 = Math.max(0, Math.min(height - 1, y0));
    const clampedX1 = Math.max(clampedX0 + 1, Math.min(width, x1));
    const clampedY1 = Math.max(clampedY0 + 1, Math.min(height, y1));

    return [
        clampedX0 / width,
        clampedY0 / height,
        (clampedX1 - clampedX0) / width,
        (clampedY1 - clampedY0) / height,
    ];
}

function extractStateBBox(state: ObjectState): number[] | null {
    if (!Array.isArray(state.points) || state.points.length < 4) {
        return null;
    }

    if (state.shapeType === ShapeType.MASK && state.points.length >= 4) {
        const [left, top, right, bottom] = state.points.slice(-4);
        return [left, top, right, bottom];
    }

    if (state.shapeType === ShapeType.RECTANGLE && state.points.length >= 4) {
        return normalizeBBox(state.points);
    }

    const xs: number[] = [];
    const ys: number[] = [];
    for (let index = 0; index < state.points.length; index += 2) {
        const x = state.points[index];
        const y = state.points[index + 1];
        if (typeof x === 'number' && typeof y === 'number') {
            xs.push(x);
            ys.push(y);
        }
    }

    if (!xs.length || !ys.length) {
        return null;
    }

    return [
        Math.min(...xs),
        Math.min(...ys),
        Math.max(...xs),
        Math.max(...ys),
    ];
}

function extractSnapshotBBox(snapshot: ShapeSnapshot): number[] | null {
    const shapeLike = {
        shapeType: snapshot.shapeType,
        points: snapshot.points,
    } as ObjectState;
    return extractStateBBox(shapeLike);
}

function arePointsInsideBBox(points: { x: number; y: number; label: number }[], bbox: number[] | null): boolean {
    if (!bbox || bbox.length !== 4) {
        return false;
    }

    const [left, top, right, bottom] = bbox;
    const positivePoints = points.filter((point) => point.label === 1);
    if (!positivePoints.length) {
        return false;
    }

    return positivePoints.every((point) => (
        point.x >= left &&
        point.x <= right &&
        point.y >= top &&
        point.y <= bottom
    ));
}

function derivePointBBox(
    points: { x: number; y: number; label: number }[],
    width: number,
    height: number,
    padding: number = 96,
): number[] | null {
    const positivePoints = points.filter((point) => point.label === 1);
    if (!positivePoints.length || width <= 0 || height <= 0) {
        return null;
    }

    const xs = positivePoints.map((point) => point.x);
    const ys = positivePoints.map((point) => point.y);
    return [
        Math.max(0, Math.floor(Math.min(...xs) - padding)),
        Math.max(0, Math.floor(Math.min(...ys) - padding)),
        Math.min(width - 1, Math.ceil(Math.max(...xs) + padding)),
        Math.min(height - 1, Math.ceil(Math.max(...ys) + padding)),
    ];
}

function buildMaskState(label: Label, frame: number, zOrder: number, maskRle: number[], source: Source): ObjectState {
    return new core.classes.ObjectState({
        objectType: ObjectType.SHAPE,
        shapeType: ShapeType.MASK,
        source,
        zOrder,
        label,
        points: maskRle,
        frame,
        occluded: false,
        attributes: {},
        descriptions: [],
    });
}

function buildShapeState(snapshot: ShapeSnapshot): ObjectState {
    return new core.classes.ObjectState({
        objectType: ObjectType.SHAPE,
        shapeType: snapshot.shapeType,
        source: snapshot.source,
        zOrder: snapshot.zOrder,
        label: snapshot.label,
        points: [...snapshot.points],
        frame: snapshot.frame,
        occluded: snapshot.occluded,
        attributes: { ...snapshot.attributes },
        descriptions: [...snapshot.descriptions],
        ...(typeof snapshot.rotation === 'number' ? { rotation: snapshot.rotation } : {}),
    });
}

function decodeMaskRleToFullMask(maskRle: number[], width: number, height: number): Uint8Array {
    const fullMask = new Uint8Array(width * height);
    if (!maskRle.length || maskRle.length < 5) {
        return fullMask;
    }

    const [left, top, right, bottom] = maskRle.slice(-4);
    const maskWidth = (right - left) + 1;
    const maskHeight = (bottom - top) + 1;
    if (maskWidth <= 0 || maskHeight <= 0) {
        return fullMask;
    }

    const localMask = core.utils.rle2Mask(maskRle.slice(0, -4), maskWidth, maskHeight);
    for (let y = 0; y < maskHeight; y++) {
        for (let x = 0; x < maskWidth; x++) {
            const localIndex = (y * maskWidth) + x;
            if (!localMask[localIndex]) {
                continue;
            }
            const imageX = left + x;
            const imageY = top + y;
            if (imageX < 0 || imageX >= width || imageY < 0 || imageY >= height) {
                continue;
            }
            fullMask[(imageY * width) + imageX] = 1;
        }
    }
    return fullMask;
}

function encodeFullMaskToCvatRle(mask: ArrayLike<number>, width: number, height: number): number[] | null {
    let left = width;
    let top = height;
    let right = -1;
    let bottom = -1;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (!mask[(y * width) + x]) {
                continue;
            }
            if (x < left) left = x;
            if (y < top) top = y;
            if (x > right) right = x;
            if (y > bottom) bottom = y;
        }
    }

    if (right < left || bottom < top) {
        return null;
    }

    const tightWidth = (right - left) + 1;
    const tightHeight = (bottom - top) + 1;
    const tightMask = new Uint8Array(tightWidth * tightHeight);
    for (let y = top; y <= bottom; y++) {
        for (let x = left; x <= right; x++) {
            tightMask[((y - top) * tightWidth) + (x - left)] = mask[(y * width) + x] ? 1 : 0;
        }
    }

    const rle = core.utils.mask2Rle(tightMask);
    return rle.length ? [...rle, left, top, right, bottom] : null;
}

function encodeBBoxToCvatMaskRle(bbox: number[] | null, width: number, height: number): number[] | null {
    if (!bbox || bbox.length !== 4 || width <= 0 || height <= 0) {
        return null;
    }

    const [x0, y0, x1, y1] = normalizeBBox(bbox);
    const left = Math.max(0, Math.floor(x0));
    const top = Math.max(0, Math.floor(y0));
    const right = Math.min(width - 1, Math.ceil(x1));
    const bottom = Math.min(height - 1, Math.ceil(y1));
    if (right < left || bottom < top) {
        return null;
    }

    const fullMask = new Uint8Array(width * height);
    for (let y = top; y <= bottom; y++) {
        for (let x = left; x <= right; x++) {
            fullMask[(y * width) + x] = 1;
        }
    }

    return encodeFullMaskToCvatRle(fullMask, width, height);
}

function countMaskPixels(mask: ArrayLike<number>): number {
    let area = 0;
    for (let index = 0; index < mask.length; index++) {
        if (mask[index]) {
            area++;
        }
    }
    return area;
}

function extractConnectedMaskCandidates(
    mask: ArrayLike<number>,
    width: number,
    height: number,
    minArea: number,
): MaterializedMaskCandidate[] {
    const totalPixels = width * height;
    if (totalPixels <= 0) {
        return [];
    }

    const binaryMask = new Uint8Array(totalPixels);
    let hasForeground = false;
    for (let index = 0; index < totalPixels; index++) {
        if (mask[index]) {
            binaryMask[index] = 1;
            hasForeground = true;
        }
    }

    if (!hasForeground) {
        return [];
    }

    const visited = new Uint8Array(totalPixels);
    const candidates: MaterializedMaskCandidate[] = [];

    for (let startIndex = 0; startIndex < totalPixels; startIndex++) {
        if (!binaryMask[startIndex] || visited[startIndex]) {
            continue;
        }

        const stack = [startIndex];
        const componentIndices: number[] = [];
        visited[startIndex] = 1;

        while (stack.length) {
            const currentIndex = stack.pop() as number;
            componentIndices.push(currentIndex);

            const x = currentIndex % width;
            const y = Math.floor(currentIndex / width);
            const neighbors = [
                x > 0 ? currentIndex - 1 : -1,
                x < width - 1 ? currentIndex + 1 : -1,
                y > 0 ? currentIndex - width : -1,
                y < height - 1 ? currentIndex + width : -1,
            ];

            for (const neighborIndex of neighbors) {
                if (neighborIndex < 0 || visited[neighborIndex] || !binaryMask[neighborIndex]) {
                    continue;
                }
                visited[neighborIndex] = 1;
                stack.push(neighborIndex);
            }
        }

        if (componentIndices.length < minArea) {
            continue;
        }

        const componentMask = new Uint8Array(totalPixels);
        for (const index of componentIndices) {
            componentMask[index] = 1;
        }
        const componentRle = encodeFullMaskToCvatRle(componentMask, width, height);
        if (componentRle) {
            candidates.push({
                rle: componentRle,
                fullMask: componentMask,
                area: componentIndices.length,
            });
        }
    }

    candidates.sort((left, right) => right.area - left.area);
    return candidates;
}

function subtractFullMasks(baseMask: ArrayLike<number>, subtractMask: ArrayLike<number>): Uint8Array {
    const remainder = new Uint8Array(baseMask.length);
    for (let index = 0; index < baseMask.length; index++) {
        if (baseMask[index] && !subtractMask[index]) {
            remainder[index] = 1;
        }
    }
    return remainder;
}

function mergeMaskCandidates(
    candidates: MaterializedMaskCandidate[],
    width: number,
    height: number,
): MaterializedMaskCandidate[] {
    if (!candidates.length) {
        return [];
    }

    const mergedMask = new Uint8Array(width * height);
    for (const candidate of candidates) {
        for (let index = 0; index < mergedMask.length; index++) {
            if (candidate.fullMask[index]) {
                mergedMask[index] = 1;
            }
        }
    }

    const mergedRle = encodeFullMaskToCvatRle(mergedMask, width, height);
    return mergedRle ? [{
        rle: mergedRle,
        fullMask: mergedMask,
        area: countMaskPixels(mergedMask),
    }] : [];
}

function maskOverlapArea(leftMask: ArrayLike<number>, rightMask: ArrayLike<number>): number {
    const totalPixels = Math.min(leftMask.length, rightMask.length);
    let overlap = 0;
    for (let index = 0; index < totalPixels; index++) {
        if (leftMask[index] && rightMask[index]) {
            overlap++;
        }
    }
    return overlap;
}

function createMaskCandidateFromRle(
    maskRle: number[] | null,
    width: number,
    height: number,
): MaterializedMaskCandidate | null {
    if (!maskRle || !maskRle.length) {
        return null;
    }

    const fullMask = decodeMaskRleToFullMask(maskRle, width, height);
    const area = countMaskPixels(fullMask);
    if (!area) {
        return null;
    }

    return {
        rle: maskRle,
        fullMask,
        area,
    };
}

function snapshotShapeState(state: ObjectState): ShapeSnapshot {
    return {
        label: state.label as Label,
        frame: state.frame,
        zOrder: state.zOrder,
        shapeType: state.shapeType,
        source: state.source as Source,
        points: [...state.points],
        occluded: state.occluded,
        attributes: { ...state.attributes },
        descriptions: [...state.descriptions],
        rotation: typeof state.rotation === 'number' ? state.rotation : null,
    };
}

function mergeClientIDs(...clientIDGroups: number[][]): number[] {
    return [...new Set(clientIDGroups.flat().filter((clientID) => Number.isInteger(clientID)))];
}

function isExpiredSAM3SessionError(errorData: unknown): boolean {
    return String(errorData).includes('Unknown or expired session_id');
}

export default function SAM3Panel(): JSX.Element {
    const dispatch = useDispatch<any>();
    const job = useSelector((state: CombinedState) => state.annotation.job.instance) as Job | null | undefined;
    const labels = useSelector((state: CombinedState) => state.annotation.job.labels);
    const states = useSelector((state: CombinedState) => state.annotation.annotations.states) as ObjectState[];
    const activatedStateID = useSelector((state: CombinedState) => state.annotation.annotations.activatedStateID);
    const activeLabelID = useSelector((state: CombinedState) => state.annotation.drawing.activeLabelID);
    const frame = useSelector((state: CombinedState) => state.annotation.player.frame.number);
    const curZOrder = useSelector((state: CombinedState) => state.annotation.annotations.zLayer.cur);
    const canvasInstance = useSelector((state: CombinedState) => state.annotation.canvas.instance) as Canvas | null;

    const [models, setModels] = useState<SAM3PanelModel[]>([]);
    const [modelsLoading, setModelsLoading] = useState(false);
    const [busy, setBusy] = useState(false);
    const [selectedModelID, setSelectedModelID] = useState<number | null>(null);
    const [selectedPromptKey, setSelectedPromptKey] = useState<string | null>(null);
    const [textPromptKeys, setTextPromptKeys] = useState<string[]>([]);
    const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.7);
    const [scoreThreshold, setScoreThreshold] = useState<number>(0.7);
    const [interactionMode, setInteractionMode] = useState<'box-positive' | 'box-negative' | 'points' | null>(null);
    const [pointSession, setPointSession] = useState<PointSessionState | null>(null);
    const [sessionState, setSessionState] = useState<SessionState | null>(null);
    const [videoTracking, setVideoTracking] = useState<VideoTrackingState | null>(null);
    const [videoFrameRange, setVideoFrameRange] = useState<[number, number]>([0, 0]);

    const sessionStateRef = useRef<SessionState | null>(null);
    const pointSessionRef = useRef<PointSessionState | null>(null);
    const ignoreCanvasCancelRef = useRef(false);
    const busyRef = useRef(false);

    useEffect(() => {
        sessionStateRef.current = sessionState;
    }, [sessionState]);

    useEffect(() => {
        pointSessionRef.current = pointSession;
    }, [pointSession]);

    useEffect(() => {
        busyRef.current = busy;
    }, [busy]);

    const eligibleLabels = useMemo(
        () => labels.filter((label) => isEligibleMaskLabel(label as Label)) as Label[],
        [labels],
    );
    const eligibleLabelsByID = useMemo(
        () => new Map(eligibleLabels.map((label) => [label.id, label])),
        [eligibleLabels],
    );

    const activeMaskLabel = useMemo(
        () => eligibleLabels.find((label) => label.id === activeLabelID) || null,
        [eligibleLabels, activeLabelID],
    );

    const selectedObject = useMemo(
        () => (activatedStateID === null ? null : states.find((state) => state.clientID === activatedStateID) || null),
        [states, activatedStateID],
    );

    const selectedMaskObject = useMemo(() => {
        if (!selectedObject || selectedObject.objectType !== ObjectType.SHAPE) {
            return null;
        }
        if (!selectedObject.label || !isEligibleMaskLabel(selectedObject.label as Label)) {
            return null;
        }
        return selectedObject;
    }, [selectedObject]);

    const currentFrameVideoPromptObjects = useMemo(() => {
        const generatedClientIDs = new Set(videoTracking?.generatedClientIDsByFrame[frame] || []);
        const seededSourceClientIDs = new Set(videoTracking?.seededSourceClientIDsByFrame[frame] || []);
        return states.filter((state) => (
            state.objectType === ObjectType.SHAPE &&
            state.frame === frame &&
            !generatedClientIDs.has(state.clientID) &&
            !seededSourceClientIDs.has(state.clientID) &&
            Boolean(state.label) &&
            isEligibleMaskLabel(state.label as Label)
        )).sort((left, right) => left.zOrder - right.zOrder);
    }, [frame, states, videoTracking]);

    const selectedVideoObjectID = useMemo(() => {
        if (!selectedMaskObject || !videoTracking) {
            return null;
        }

        return (
            videoTracking.generatedObjectIDsByFrame[frame]?.[selectedMaskObject.clientID] ??
            videoTracking.seededObjectIDsByFrame[frame]?.[selectedMaskObject.clientID] ??
            null
        );
    }, [frame, selectedMaskObject, videoTracking]);

    const groupedModels = useMemo(() => {
        const groups = new Map<string, SAM3PanelModel[]>();
        for (const model of models) {
            const current = groups.get(model.procedure_type) || [];
            current.push(model);
            groups.set(model.procedure_type, current);
        }
        return [...groups.entries()];
    }, [models]);

    const modelOptions = useMemo(
        () => groupedModels.map(([procedureType, procedureModels]) => ({
            label: procedureType,
            options: procedureModels.map((model) => ({
                label: model.display_name,
                value: model.id,
            })),
        })),
        [groupedModels],
    );

    const selectedModel = useMemo(
        () => models.find((model) => model.id === selectedModelID) || null,
        [models, selectedModelID],
    );

    const promptBindings = useMemo<PromptBinding[]>(() => {
        const promptOptions = selectedModel?.prompt_options?.length ? selectedModel.prompt_options : eligibleLabels.map((label) => ({
            key: label.name,
            value: label.name,
            display_name: label.name,
            prompt: label.name,
            color: null,
            auto_mask_priority: null,
            component_mode: 'union' as const,
            min_component_area: DEFAULT_UNION_MIN_COMPONENT_AREA,
        }));

        return promptOptions.map((option) => {
            const candidateTokens = new Set(
                [option.display_name, option.prompt, option.value, option.key]
                    .filter(Boolean)
                    .map((value) => normalizePromptToken(String(value))),
            );
            const matchedLabel = eligibleLabels.find((label) => candidateTokens.has(normalizePromptToken(label.name))) || null;
            return {
                option,
                label: matchedLabel,
            };
        });
    }, [eligibleLabels, selectedModel]);

    const promptOptions = useMemo(
        () => promptBindings.map(({ option, label }) => ({
            label: label ? option.display_name : `${option.display_name} (no CVAT label)`,
            value: option.key,
        })),
        [promptBindings],
    );

    const textPromptOptions = useMemo(
        () => promptBindings.map(({ option, label }) => ({
            label: label ? option.display_name : `${option.display_name} (no CVAT label)`,
            value: option.key,
            disabled: !label,
        })),
        [promptBindings],
    );

    const selectedPromptBinding = useMemo(
        () => promptBindings.find((binding) => binding.option.key === selectedPromptKey) || null,
        [promptBindings, selectedPromptKey],
    );

    const activePromptBinding = useMemo(
        () => (activeMaskLabel ? promptBindings.find((binding) => binding.label?.id === activeMaskLabel.id) || null : null),
        [activeMaskLabel, promptBindings],
    );

    const selectedObjectPromptBinding = useMemo(
        () => (
            selectedMaskObject ?
                promptBindings.find((binding) => binding.label?.id === selectedMaskObject.label.id) || null :
                null
        ),
        [promptBindings, selectedMaskObject],
    );

    const selectedPromptOverridesSelectedObject = useMemo(() => Boolean(
        selectedMaskObject &&
        selectedPromptBinding?.label &&
        selectedPromptBinding.label.id !== selectedMaskObject.label.id,
    ), [selectedMaskObject, selectedPromptBinding]);

    const unmappedPromptBindings = useMemo(
        () => promptBindings.filter((binding) => !binding.label),
        [promptBindings],
    );

    const mappedPromptBindings = useMemo(
        () => promptBindings.filter((binding) => !!binding.label),
        [promptBindings],
    );

    const getPromptOptionForLabel = useCallback((label: Label): SAM3PromptOption | null => (
        promptBindings.find((binding) => binding.label?.id === label.id)?.option || null
    ), [promptBindings]);

    const materializeMaskCandidatesForLabel = useCallback((
        label: Label,
        maskRle: number[] | null,
        width: number,
        height: number,
    ): MaterializedMaskCandidate[] => {
        if (!maskRle || !maskRle.length) {
            return [];
        }

        const option = getPromptOptionForLabel(label);
        const componentMode = option?.component_mode || 'union';
        const minComponentArea = option?.min_component_area || (
            componentMode === 'components' ? DEFAULT_SPLIT_MIN_COMPONENT_AREA : DEFAULT_UNION_MIN_COMPONENT_AREA
        );
        const fullMask = decodeMaskRleToFullMask(maskRle, width, height);
        const candidates = extractConnectedMaskCandidates(fullMask, width, height, minComponentArea);
        if (!candidates.length) {
            return [];
        }

        if (componentMode === 'components') {
            return candidates;
        }

        return mergeMaskCandidates(candidates, width, height);
    }, [getPromptOptionForLabel]);

    const pickPrimaryMaskCandidateIndex = useCallback((
        candidates: MaterializedMaskCandidate[],
        targetState: ObjectState | ShapeSnapshot | null,
        width: number,
        height: number,
    ): number => {
        if (!candidates.length) {
            return -1;
        }

        const referencePoints = targetState && Array.isArray(targetState.points) ? targetState.points : null;
        if (!referencePoints || referencePoints.length < 5) {
            return 0;
        }

        const referenceMask = decodeMaskRleToFullMask(referencePoints, width, height);
        let bestIndex = 0;
        let bestOverlap = -1;
        let bestArea = -1;
        for (const [index, candidate] of candidates.entries()) {
            const overlap = maskOverlapArea(referenceMask, candidate.fullMask);
            if (overlap > bestOverlap || (overlap === bestOverlap && candidate.area > bestArea)) {
                bestIndex = index;
                bestOverlap = overlap;
                bestArea = candidate.area;
            }
        }

        return bestIndex;
    }, []);

    const refreshAnnotations = useCallback(async () => {
        await dispatch(fetchAnnotationsAsync());
    }, [dispatch]);

    const loadCurrentFrameStates = useCallback(async (): Promise<ObjectState[]> => {
        if (!job) return [];
        return job.annotations.get(frame, false, []);
    }, [job, frame]);

    const findObjectState = useCallback(async (clientID: number): Promise<ObjectState | null> => {
        const currentStates = await loadCurrentFrameStates();
        return currentStates.find((state) => state.clientID === clientID) || null;
    }, [loadCurrentFrameStates]);

    const invalidateSession = useCallback((sessionID?: string | null): void => {
        if (!sessionID || sessionStateRef.current?.sessionID === sessionID) {
            sessionStateRef.current = null;
            setSessionState(null);
        }

        if (pointSessionRef.current?.logitsToken) {
            const nextPointSession = {
                ...pointSessionRef.current,
                logitsToken: null,
            };
            pointSessionRef.current = nextPointSession;
            setPointSession(nextPointSession);
        }
    }, []);

    const destroySession = useCallback(async (sessionOverride?: SessionState | null): Promise<void> => {
        const currentSession = sessionOverride ?? sessionStateRef.current;
        if (!job || !currentSession) return;

        try {
            await serverProxy.jobs.deleteSAM3Session(job.id, currentSession.sessionID, currentSession.modelID);
        } catch (errorData: unknown) {
            if (!isExpiredSAM3SessionError(errorData)) {
                notification.warning({
                    message: 'Failed to delete SAM3 session',
                    description: String(errorData),
                });
            }
        } finally {
            invalidateSession(currentSession.sessionID);
        }
    }, [invalidateSession, job]);

    const endCanvasInteraction = useCallback(() => {
        if (!canvasInstance) return;
        ignoreCanvasCancelRef.current = true;
        canvasInstance.interact({ enabled: false });
        canvasInstance.cancel();
        setInteractionMode(null);
    }, [canvasInstance]);

    const finishPointSession = useCallback(async (revert = false): Promise<void> => {
        const current = pointSessionRef.current;
        if (!current) {
            endCanvasInteraction();
            return;
        }

        setBusy(true);
        try {
            if (revert && job) {
                for (const clientID of current.auxiliaryClientIDs) {
                    const auxiliaryState = await findObjectState(clientID);
                    if (auxiliaryState) {
                        await auxiliaryState.delete(frame, false);
                    }
                }
                if (current.createdDuringSession && current.targetClientID !== null) {
                    const createdState = await findObjectState(current.targetClientID);
                    if (createdState) {
                        await createdState.delete(frame, false);
                    }
                } else if (current.replacedExistingObject && current.originalState) {
                    if (current.targetClientID !== null) {
                        const replacementState = await findObjectState(current.targetClientID);
                        if (replacementState) {
                            await replacementState.delete(frame, false);
                        }
                    }
                    await job.annotations.put([buildShapeState(current.originalState)]);
                } else if (current.targetClientID !== null && current.originalState) {
                    const targetState = await findObjectState(current.targetClientID);
                    if (targetState) {
                        targetState.points = [...current.originalState.points];
                        await targetState.save();
                    }
                }
                await refreshAnnotations();
            }
        } catch (errorData: unknown) {
            notification.error({
                message: 'Failed to close point session',
                description: String(errorData),
            });
        } finally {
            pointSessionRef.current = null;
            setPointSession(null);
            setBusy(false);
            endCanvasInteraction();
        }
    }, [endCanvasInteraction, findObjectState, frame, job, refreshAnnotations]);

    const ensureSession = useCallback(async (): Promise<SessionState> => {
        if (!job) {
            throw new Error('No open job');
        }
        if (!selectedModelID) {
            throw new Error('Select a SAM3 model first');
        }

        const currentSession = sessionStateRef.current;
        if (currentSession &&
            currentSession.frame === frame &&
            currentSession.modelID === selectedModelID &&
            currentSession.confidenceThreshold === confidenceThreshold) {
            return currentSession;
        }

        if (currentSession) {
            await destroySession(currentSession);
        }

        const created = await serverProxy.jobs.createSAM3Session(job.id, {
            model_id: selectedModelID,
            frame,
            confidence_threshold: confidenceThreshold,
        });
        const nextSession = {
            sessionID: created.session_id,
            frame,
            modelID: selectedModelID,
            width: created.width,
            height: created.height,
            confidenceThreshold,
        };
        sessionStateRef.current = nextSession;
        setSessionState(nextSession);
        return nextSession;
    }, [confidenceThreshold, destroySession, frame, job, selectedModelID]);

    const withSAM3SessionRetry = useCallback(async <T,>(
        run: (session: SessionState, isRetry: boolean) => Promise<T>,
    ): Promise<T> => {
        let session = await ensureSession();
        try {
            return await run(session, false);
        } catch (errorData: unknown) {
            if (!isExpiredSAM3SessionError(errorData)) {
                throw errorData;
            }

            invalidateSession(session.sessionID);
            session = await ensureSession();
            return run(session, true);
        }
    }, [ensureSession, invalidateSession]);

    const resolveInitialPointMaskRle = useCallback(async (
        currentPointSession: PointSessionState,
        targetLabel: Label,
    ): Promise<number[] | null> => {
        if (currentPointSession.logitsToken) {
            return null;
        }

        const candidateClientIDs = [
            currentPointSession.targetClientID,
            currentPointSession.sourceClientID,
        ].filter((clientID): clientID is number => clientID !== null);

        for (const clientID of candidateClientIDs) {
            const currentState = await findObjectState(clientID);
            if (
                currentState &&
                currentState.shapeType === ShapeType.MASK &&
                currentState.label?.id === targetLabel.id &&
                Array.isArray(currentState.points) &&
                currentState.points.length >= 5
            ) {
                return [...currentState.points];
            }
        }

        if (
            currentPointSession.originalState?.shapeType === ShapeType.MASK &&
            currentPointSession.originalState.label.id === targetLabel.id
        ) {
            return [...currentPointSession.originalState.points];
        }

        return null;
    }, [findObjectState]);

    const putMaskObject = useCallback(async (
        label: Label,
        maskRle: number[],
        source: Source,
        zOrderOverride?: number,
    ): Promise<MaskMutationResult> => {
        if (!job) throw new Error('No open job');
        const currentSession = sessionStateRef.current;
        const candidates = currentSession ?
            materializeMaskCandidatesForLabel(label, maskRle, currentSession.width, currentSession.height) :
            [];
        const effectiveCandidates = candidates.length ? candidates : (
            currentSession ?
                [createMaskCandidateFromRle(maskRle, currentSession.width, currentSession.height)].filter(Boolean) as MaterializedMaskCandidate[] :
                [{ rle: maskRle, fullMask: new Uint8Array(), area: 0 }]
        );
        const statesToCreate = effectiveCandidates.map((candidate) => buildMaskState(
            label,
            frame,
            zOrderOverride ?? curZOrder,
            candidate.rle,
            source,
        ));
        const [primaryClientID, ...auxiliaryClientIDs] = await job.annotations.put(statesToCreate);
        await refreshAnnotations();
        dispatch(activateObject(primaryClientID, null, null));
        return {
            primaryClientID,
            auxiliaryClientIDs,
        };
    }, [curZOrder, dispatch, frame, job, materializeMaskCandidatesForLabel, refreshAnnotations]);

    const computeRemainderMaskRles = useCallback((
        targetState: ObjectState,
        subtractMask: ArrayLike<number>,
        width: number,
        height: number,
    ): number[][] => {
        if (targetState.shapeType !== ShapeType.MASK || !Array.isArray(targetState.points) || targetState.points.length < 5) {
            return [];
        }

        const originalMask = decodeMaskRleToFullMask(targetState.points, width, height);
        const remainderMask = subtractFullMasks(originalMask, subtractMask);
        const remainderRle = encodeFullMaskToCvatRle(remainderMask, width, height);
        if (!remainderRle) {
            return [];
        }
        return materializeMaskCandidatesForLabel(
            targetState.label as Label,
            remainderRle,
            width,
            height,
        ).map((candidate) => candidate.rle);
    }, [materializeMaskCandidatesForLabel]);

    const updateMaskObject = useCallback(async (
        targetState: ObjectState,
        maskRle: number[],
        source: Source,
    ): Promise<MaskMutationResult> => {
        if (targetState.shapeType === ShapeType.MASK) {
            const currentSession = sessionStateRef.current;
            if (!currentSession) {
                targetState.points = [...maskRle];
                await targetState.save();
                await refreshAnnotations();
                dispatch(activateObject(targetState.clientID, null, null));
                return {
                    primaryClientID: targetState.clientID as number,
                    auxiliaryClientIDs: [],
                };
            }
            const candidates = currentSession ?
                materializeMaskCandidatesForLabel(
                    targetState.label as Label,
                    maskRle,
                    currentSession.width,
                    currentSession.height,
                ) :
                [];
            const fallbackCandidate = currentSession ?
                createMaskCandidateFromRle(maskRle, currentSession.width, currentSession.height) :
                null;
            const effectiveCandidates = candidates.length ? candidates : (fallbackCandidate ? [fallbackCandidate] : []);
            if (!effectiveCandidates.length) {
                throw new Error('SAM3 mask became empty after component filtering');
            }

            const primaryIndex = currentSession ?
                pickPrimaryMaskCandidateIndex(effectiveCandidates, targetState, currentSession.width, currentSession.height) :
                0;
            const [primaryCandidate] = effectiveCandidates.splice(primaryIndex, 1);
            const mergedCandidates = currentSession ? mergeMaskCandidates(
                [primaryCandidate, ...effectiveCandidates],
                currentSession.width,
                currentSession.height,
            ) : [primaryCandidate];
            const unionMask = mergedCandidates[0]?.fullMask || primaryCandidate.fullMask;
            const remainderMaskRles = currentSession ?
                computeRemainderMaskRles(targetState, unionMask, currentSession.width, currentSession.height) :
                [];

            targetState.points = [...primaryCandidate.rle];
            await targetState.save();
            if (job && (effectiveCandidates.length || remainderMaskRles.length)) {
                const siblingStates = [
                    ...effectiveCandidates.map((candidate) => buildMaskState(
                        targetState.label as Label,
                        frame,
                        targetState.zOrder,
                        candidate.rle,
                        targetState.source as Source,
                    )),
                    ...remainderMaskRles.map((remainderMaskRle) => buildMaskState(
                        targetState.label as Label,
                        frame,
                        targetState.zOrder,
                        remainderMaskRle,
                        targetState.source as Source,
                    )),
                ];
                if (siblingStates.length) {
                    const siblingClientIDs = await job.annotations.put(siblingStates);
                    await refreshAnnotations();
                    dispatch(activateObject(targetState.clientID, null, null));
                    return {
                        primaryClientID: targetState.clientID as number,
                        auxiliaryClientIDs: siblingClientIDs,
                    };
                }
            }
            await refreshAnnotations();
            dispatch(activateObject(targetState.clientID, null, null));
            return {
                primaryClientID: targetState.clientID as number,
                auxiliaryClientIDs: [],
            };
        }

        await targetState.delete(frame, false);
        return putMaskObject(targetState.label as Label, maskRle, source, targetState.zOrder);
    }, [computeRemainderMaskRles, dispatch, frame, job, materializeMaskCandidatesForLabel, pickPrimaryMaskCandidateIndex, putMaskObject, refreshAnnotations]);

    const replaceMaskObjectLabel = useCallback(async (
        targetState: ObjectState,
        label: Label,
        maskRle: number[],
        source: Source,
    ): Promise<MaskMutationResult> => {
        if (targetState.shapeType === ShapeType.MASK && targetState.label.id === label.id) {
            return updateMaskObject(targetState, maskRle, source);
        }

        const zOrder = targetState.zOrder;
        const currentSession = sessionStateRef.current;
        if (!currentSession) {
            await targetState.delete(frame, false);
            const state = buildMaskState(label, frame, zOrder, maskRle, source);
            const [primaryClientID] = await job.annotations.put([state]);
            await refreshAnnotations();
            dispatch(activateObject(primaryClientID, null, null));
            return {
                primaryClientID,
                auxiliaryClientIDs: [],
            };
        }
        const candidates = currentSession ?
            materializeMaskCandidatesForLabel(label, maskRle, currentSession.width, currentSession.height) :
            [];
        const fallbackCandidate = currentSession ?
            createMaskCandidateFromRle(maskRle, currentSession.width, currentSession.height) :
            null;
        const effectiveCandidates = candidates.length ? candidates : (fallbackCandidate ? [fallbackCandidate] : []);
        if (!effectiveCandidates.length) {
            throw new Error('SAM3 mask became empty after component filtering');
        }
        const primaryIndex = currentSession ?
            pickPrimaryMaskCandidateIndex(effectiveCandidates, targetState, currentSession.width, currentSession.height) :
            0;
        const [primaryCandidate] = effectiveCandidates.splice(primaryIndex, 1);
        const mergedCandidates = currentSession ? mergeMaskCandidates(
            [primaryCandidate, ...effectiveCandidates],
            currentSession.width,
            currentSession.height,
        ) : [primaryCandidate];
        const unionMask = mergedCandidates[0]?.fullMask || primaryCandidate.fullMask;
        const remainderMaskRles = currentSession ?
            computeRemainderMaskRles(targetState, unionMask, currentSession.width, currentSession.height) :
            [];
        await targetState.delete(frame, false);
        const replacementStates = [
            buildMaskState(label, frame, zOrder, primaryCandidate.rle, source),
            ...effectiveCandidates.map((candidate) => buildMaskState(
                label,
                frame,
                zOrder,
                candidate.rle,
                source,
            )),
            ...remainderMaskRles.map((remainderMaskRle) => buildMaskState(
                targetState.label as Label,
                frame,
                zOrder,
                remainderMaskRle,
                targetState.source as Source,
            )),
        ];
        const [primaryClientID, ...auxiliaryClientIDs] = await job.annotations.put(replacementStates);
        await refreshAnnotations();
        dispatch(activateObject(primaryClientID, null, null));
        return {
            primaryClientID,
            auxiliaryClientIDs,
        };
    }, [computeRemainderMaskRles, dispatch, frame, job, materializeMaskCandidatesForLabel, pickPrimaryMaskCandidateIndex, refreshAnnotations, updateMaskObject]);

    const loadModels = useCallback(async () => {
        if (!job) return;
        setModelsLoading(true);
        try {
            const response = await serverProxy.jobs.getSAM3Models(job.id);
            setModels(response);
            setSelectedModelID((current) => {
                if (current && response.some((model) => model.id === current)) {
                    return current;
                }
                return response.length ? response[0].id : null;
            });
        } catch (errorData: unknown) {
            notification.error({
                message: 'Failed to load SAM3 models',
                description: String(errorData),
            });
        } finally {
            setModelsLoading(false);
        }
    }, [job]);

    useEffect(() => {
        void loadModels();
    }, [loadModels]);

    useEffect(() => {
        const nextPromptKeys = promptBindings.map((binding) => binding.option.key);
        const mappedPromptKeys = promptBindings
            .filter((binding) => !!binding.label)
            .map((binding) => binding.option.key);
        setTextPromptKeys((current) => {
            const filtered = current.filter((key) => nextPromptKeys.includes(key));
            if (filtered.length) {
                return filtered;
            }
            if (mappedPromptKeys.length) {
                return mappedPromptKeys;
            }
            return [];
        });
        setSelectedPromptKey((current) => {
            if (current && nextPromptKeys.includes(current)) {
                return current;
            }
            const defaultPromptKey = selectedModel?.default_prompt_key;
            if (defaultPromptKey && nextPromptKeys.includes(defaultPromptKey)) {
                return defaultPromptKey;
            }
            return nextPromptKeys[0] || null;
        });
    }, [promptBindings, selectedModel]);

    useEffect(() => {
        if (selectedModel) {
            setConfidenceThreshold(selectedModel.confidence_threshold);
            setScoreThreshold(selectedModel.score_threshold);
        }
    }, [selectedModel?.id]);

    useEffect(() => {
        const currentSession = sessionStateRef.current;
        if (!currentSession) return;
        if (!job ||
            currentSession.frame !== frame ||
            currentSession.modelID !== selectedModelID ||
            currentSession.confidenceThreshold !== confidenceThreshold) {
            void destroySession(currentSession);
        }
    }, [confidenceThreshold, destroySession, frame, job, selectedModelID]);

    useEffect(() => {
        if (!pointSessionRef.current) return;
        pointSessionRef.current = null;
        setPointSession(null);
        endCanvasInteraction();
    }, [endCanvasInteraction, frame, job?.id, selectedModelID]);

    useEffect(() => {
        const currentPointSession = pointSessionRef.current;
        if (!currentPointSession) return;
        if (currentPointSession.targetClientID !== null && currentPointSession.targetClientID !== activatedStateID) {
            void finishPointSession(false);
        }
    }, [activatedStateID, finishPointSession]);

    useEffect(() => () => {
        void destroySession(sessionStateRef.current);
    }, [destroySession]);

    const resolvePromptTarget = useCallback((options?: PromptTargetOptions): { label: Label; prompt: string; promptDisplay: string } => {
        const preferSelectedPrompt = options?.preferSelectedPrompt ?? false;

        if (preferSelectedPrompt && selectedPromptBinding?.label) {
            return {
                label: selectedPromptBinding.label,
                prompt: selectedPromptBinding.option.prompt,
                promptDisplay: selectedPromptBinding.option.display_name,
            };
        }

        if (selectedMaskObject) {
            const label = selectedMaskObject.label as Label;
            return {
                label,
                prompt: selectedObjectPromptBinding?.option.prompt || label.name,
                promptDisplay: selectedObjectPromptBinding?.option.display_name || label.name,
            };
        }

        if (selectedPromptBinding?.label) {
            return {
                label: selectedPromptBinding.label,
                prompt: selectedPromptBinding.option.prompt,
                promptDisplay: selectedPromptBinding.option.display_name,
            };
        }

        if (activePromptBinding?.label) {
            return {
                label: activePromptBinding.label,
                prompt: activePromptBinding.option.prompt,
                promptDisplay: activePromptBinding.option.display_name,
            };
        }

        if (selectedPromptBinding && !selectedPromptBinding.label) {
            throw new Error(`No matching CVAT label exists for "${selectedPromptBinding.option.display_name}"`);
        }

        if (!selectedModel?.prompt_options?.length && activeMaskLabel) {
            return {
                label: activeMaskLabel,
                prompt: activeMaskLabel.name,
                promptDisplay: activeMaskLabel.name,
            };
        }

        throw new Error('Select an anatomy prompt with a matching CVAT label or select an existing object');
    }, [
        activeMaskLabel,
        activePromptBinding,
        selectedMaskObject,
        selectedModel,
        selectedObjectPromptBinding,
        selectedPromptBinding,
    ]);

    const hasPromptTarget = useMemo(() => {
        if (selectedMaskObject) {
            return true;
        }
        if (selectedPromptBinding?.label || activePromptBinding?.label) {
            return true;
        }
        return !selectedModel?.prompt_options?.length && !!activeMaskLabel;
    }, [
        activeMaskLabel,
        activePromptBinding,
        selectedMaskObject,
        selectedModel,
        selectedPromptBinding,
    ]);

    const getFullFrameBBox = useCallback(async (): Promise<number[]> => {
        const currentSession = await ensureSession();
        return [0, 0, currentSession.width - 1, currentSession.height - 1];
    }, [ensureSession]);

    const syncPromptLabels = useCallback(async () => {
        if (!job || !selectedModelID) return;

        setBusy(true);
        try {
            const result = await serverProxy.jobs.syncSAM3Labels(job.id, {
                model_id: selectedModelID,
            });

            if (result.conflicts.length) {
                notification.warning({
                    message: 'Some anatomy labels could not be synced',
                    description: result.conflicts
                        .map((conflict) => `${conflict.prompt_display_name} -> ${conflict.label_name} (${conflict.label_type})`)
                        .join(', '),
                });
            }

            if (result.reload_required) {
                notification.success({
                    message: 'Anatomy labels created',
                    description: `Created ${result.created_labels.length} labels. Reloading the annotation page to activate them.`,
                });
                window.location.reload();
                return;
            }

            notification.info({
                message: 'Anatomy labels already exist',
                description: result.existing_labels.length ?
                    `Using existing labels: ${result.existing_labels.join(', ')}` :
                    'No new labels were required.',
            });
        } catch (errorData: unknown) {
            notification.error({
                message: 'Failed to create anatomy labels',
                description: String(errorData),
            });
        } finally {
            setBusy(false);
        }
    }, [job, selectedModelID]);

    const replaceAutoMasks = useCallback(async (
        maskResults: { label: Label; maskRle: number[] | null }[],
        width: number,
        height: number,
    ): Promise<number> => {
        if (!job) {
            throw new Error('No open job');
        }

        const labelsToReplace = new Set(maskResults.map((result) => result.label.name));

        const existingAutoMasks = states.filter((state) => (
            state.frame === frame &&
            state.objectType === ObjectType.SHAPE &&
            state.source === Source.AUTO &&
            labelsToReplace.has(state.label.name)
        ));

        for (const state of existingAutoMasks) {
            await state.delete(frame, false);
        }

        const objectsToCreate = maskResults.flatMap((result) => {
            if (!Array.isArray(result.maskRle) || !result.maskRle.length) {
                return [];
            }

            return materializeMaskCandidatesForLabel(
                result.label,
                result.maskRle,
                width,
                height,
            ).map((candidate) => buildMaskState(
                result.label,
                frame,
                curZOrder,
                candidate.rle,
                Source.AUTO,
            ));
        });

        if (objectsToCreate.length) {
            await job.annotations.put(objectsToCreate);
            await refreshAnnotations();
        }

        return objectsToCreate.length;
    }, [curZOrder, frame, job, materializeMaskCandidatesForLabel, refreshAnnotations, states]);

    const applyExclusiveAutoMaskControl = useCallback((
        maskResults: AutoMaskResult[],
        width: number,
        height: number,
    ): { label: Label; maskRle: number[] | null }[] => {
        const labelsToReplace = new Set(maskResults.map((result) => result.label.name));
        const claimedPixels = new Uint8Array(width * height);

        for (const state of states) {
            if (state.frame !== frame ||
                state.objectType !== ObjectType.SHAPE ||
                state.shapeType !== ShapeType.MASK ||
                !Array.isArray(state.points) ||
                state.points.length < 5) {
                continue;
            }
            if (state.source === Source.AUTO && labelsToReplace.has(state.label.name)) {
                continue;
            }

            const existingMask = decodeMaskRleToFullMask(state.points, width, height);
            for (let index = 0; index < existingMask.length; index++) {
                if (existingMask[index]) {
                    claimedPixels[index] = 1;
                }
            }
        }

        const rankedResults = maskResults.map((result, originalIndex) => {
            const fullMask = result.maskRle ? decodeMaskRleToFullMask(result.maskRle, width, height) : new Uint8Array(width * height);
            let area = 0;
            for (let index = 0; index < fullMask.length; index++) {
                if (fullMask[index]) {
                    area++;
                }
            }
            return {
                ...result,
                fullMask,
                area,
                originalIndex,
            };
        });

        rankedResults.sort((leftResult, rightResult) => (
            leftResult.autoMaskPriority - rightResult.autoMaskPriority ||
            rightResult.confidence - leftResult.confidence ||
            leftResult.area - rightResult.area ||
            leftResult.label.name.localeCompare(rightResult.label.name) ||
            leftResult.originalIndex - rightResult.originalIndex
        ));

        const exclusiveMasksByIndex = new Map<number, number[] | null>();
        for (const result of rankedResults) {
            if (!result.maskRle || !result.area) {
                exclusiveMasksByIndex.set(result.originalIndex, null);
                continue;
            }

            const exclusiveMask = new Uint8Array(width * height);
            let hasPixels = false;
            for (let index = 0; index < result.fullMask.length; index++) {
                if (result.fullMask[index] && !claimedPixels[index]) {
                    claimedPixels[index] = 1;
                    exclusiveMask[index] = 1;
                    hasPixels = true;
                }
            }

            exclusiveMasksByIndex.set(
                result.originalIndex,
                hasPixels ? encodeFullMaskToCvatRle(exclusiveMask, width, height) : null,
            );
        }

        return maskResults.map((result, index) => ({
            label: result.label,
            maskRle: exclusiveMasksByIndex.get(index) ?? null,
        }));
    }, [frame, states]);

    const runTextPrelabel = useCallback(async () => {
        if (!job || !selectedModelID) return;
        if (!textPromptKeys.length) {
            notification.warning({ message: 'Select at least one anatomy prompt for text prelabel' });
            return;
        }

        setBusy(true);
        try {
            const executableBindings = promptBindings.filter((binding) => (
                textPromptKeys.includes(binding.option.key) && binding.label
            ));
            const skippedBindings = promptBindings.filter((binding) => (
                textPromptKeys.includes(binding.option.key) && !binding.label
            ));

            if (!executableBindings.length) {
                throw new Error('No selected anatomy prompts have matching CVAT labels');
            }

            const { session, results } = await withSAM3SessionRetry(async (activeSession) => ({
                session: activeSession,
                results: await serverProxy.jobs.inferSAM3Text(job.id, {
                    model_id: selectedModelID,
                    session_id: activeSession.sessionID,
                    labels: executableBindings.map((binding) => binding.option.prompt),
                    score_threshold: scoreThreshold,
                }) as SerializedSAM3TextResult[],
            }));

            const createdCount = await replaceAutoMasks(
                results.map((result) => {
                    const binding = executableBindings.find(
                        (item) => normalizePromptToken(item.option.prompt) === normalizePromptToken(result.label_name),
                    );
                    return {
                        label: binding?.label as Label,
                        maskRle: result.mask_rle,
                    };
                }).filter((result) => !!result.label),
                session.width,
                session.height,
            );

            if (!createdCount) {
                notification.info({ message: 'SAM3 did not return any text masks for the selected labels' });
            }

            if (skippedBindings.length) {
                notification.warning({
                    message: 'Skipped prompts without CVAT labels',
                    description: skippedBindings.map((binding) => binding.option.display_name).join(', '),
                });
            }
        } catch (errorData: unknown) {
            notification.error({
                message: 'SAM3 text prelabel failed',
                description: String(errorData),
            });
        } finally {
            setBusy(false);
        }
    }, [
        ensureSession,
        job,
        promptBindings,
        replaceAutoMasks,
        refreshAnnotations,
        scoreThreshold,
        selectedModelID,
        textPromptKeys,
        withSAM3SessionRetry,
    ]);

    const runFullImageAutoInference = useCallback(async () => {
        if (!job || !selectedModelID) return;

        const executableBindings = promptBindings.filter((binding) => (
            textPromptKeys.includes(binding.option.key) && binding.label
        ));
        if (!executableBindings.length) {
            notification.warning({ message: 'Select at least one mapped anatomy prompt for full-image auto inference' });
            return;
        }

        setBusy(true);
        try {
            const { session, results } = await withSAM3SessionRetry(async (activeSession) => ({
                session: activeSession,
                results: await serverProxy.jobs.inferSAM3Text(job.id, {
                    model_id: selectedModelID,
                    session_id: activeSession.sessionID,
                    labels: executableBindings.map((binding) => binding.option.prompt),
                    score_threshold: scoreThreshold,
                }) as SerializedSAM3TextResult[],
            }));

            const resultsByPrompt = new Map(
                results.map((result) => [normalizePromptToken(result.label_name), result]),
            );

            const createdCount = await replaceAutoMasks(
                applyExclusiveAutoMaskControl(
                    executableBindings.map((binding, index) => {
                        const result = resultsByPrompt.get(normalizePromptToken(binding.option.prompt));
                        return {
                            label: binding.label as Label,
                            maskRle: result?.mask_rle || null,
                            confidence: result?.confidence || 0,
                            autoMaskPriority: binding.option.auto_mask_priority ?? ((index + 1) * 100),
                        };
                    }),
                    session.width,
                    session.height,
                ),
                session.width,
                session.height,
            );
            if (!createdCount) {
                notification.info({ message: 'SAM3 did not return any masks for the selected full-image prompts' });
            }
        } catch (errorData: unknown) {
            notification.error({
                message: 'SAM3 full-image auto inference failed',
                description: String(errorData),
            });
        } finally {
            setBusy(false);
        }
    }, [
        ensureSession,
        job,
        promptBindings,
        applyExclusiveAutoMaskControl,
        replaceAutoMasks,
        scoreThreshold,
        selectedModelID,
        textPromptKeys,
        withSAM3SessionRetry,
    ]);

    const runBoxInference = useCallback(async (negative: boolean, bbox: number[]) => {
        if (!job || !selectedModelID) return;
        if (negative && !selectedMaskObject) {
            notification.warning({ message: 'Negative box refinement requires a selected object' });
            return;
        }

        setBusy(true);
        try {
            const target = resolvePromptTarget({
                preferSelectedPrompt: !negative && selectedPromptOverridesSelectedObject,
            });
            const result = await withSAM3SessionRetry(async (activeSession) => (
                serverProxy.jobs.inferSAM3Box(job.id, {
                    model_id: selectedModelID,
                    session_id: activeSession.sessionID,
                    label_name: target.label.name,
                    prompt: target.prompt,
                    bbox,
                    negative,
                    score_threshold: scoreThreshold,
                }) as SerializedSAM3MaskResult
            ));

            if (!result.mask_rle) {
                notification.info({ message: 'SAM3 did not return a mask for this box prompt' });
                return;
            }

            if (selectedMaskObject && !selectedPromptOverridesSelectedObject) {
                const targetState = await findObjectState(selectedMaskObject.clientID);
                if (!targetState) {
                    throw new Error('Selected object is no longer available');
                }
                await updateMaskObject(targetState, result.mask_rle, Source.SEMI_AUTO);
            } else {
                await putMaskObject(target.label, result.mask_rle, Source.SEMI_AUTO);
            }
        } catch (errorData: unknown) {
            notification.error({
                message: 'SAM3 box inference failed',
                description: String(errorData),
            });
        } finally {
            setBusy(false);
        }
    }, [
        ensureSession,
        findObjectState,
        job,
        putMaskObject,
        resolvePromptTarget,
        selectedPromptOverridesSelectedObject,
        scoreThreshold,
        selectedMaskObject,
        selectedModelID,
        updateMaskObject,
        withSAM3SessionRetry,
    ]);

    const runFullImageInference = useCallback(async () => {
        try {
            const bbox = await getFullFrameBBox();
            await runBoxInference(false, bbox);
        } catch (errorData: unknown) {
            notification.error({
                message: 'SAM3 full-image inference failed',
                description: String(errorData),
            });
        }
    }, [getFullFrameBBox, runBoxInference]);

    const runPointInference = useCallback(async (points: { x: number; y: number; label: number }[]) => {
        if (!job || !selectedModelID) return;

        const currentPointSession = pointSessionRef.current || {
            targetClientID: selectedPromptOverridesSelectedObject ? null : (selectedMaskObject?.clientID ?? null),
            sourceClientID: selectedMaskObject?.clientID ?? null,
            auxiliaryClientIDs: [],
            createdDuringSession: false,
            replacedExistingObject: false,
            originalState: selectedMaskObject ? snapshotShapeState(selectedMaskObject) : null,
            logitsToken: null,
            mode: 1 as PointMode,
        };
        const preferSelectedPrompt = currentPointSession.mode === 1 && selectedPromptOverridesSelectedObject;

        let targetLabel: Label;
        let targetPrompt: string;
        try {
            const target = resolvePromptTarget({
                preferSelectedPrompt,
            });
            targetLabel = target.label;
            targetPrompt = target.prompt;
        } catch (errorData: unknown) {
            notification.warning({ message: String(errorData) });
            return;
        }

        setBusy(true);
        try {
            let activePointSession = currentPointSession;
            const { result, localPointBBox } = await withSAM3SessionRetry(async (activeSession, isRetry) => {
                if (isRetry && activePointSession.logitsToken) {
                    activePointSession = {
                        ...activePointSession,
                        logitsToken: null,
                    };
                    pointSessionRef.current = activePointSession;
                    setPointSession(activePointSession);
                }

                const nextLocalPointBBox = derivePointBBox(points, activeSession.width, activeSession.height);
                const initialMaskRle = await resolveInitialPointMaskRle(activePointSession, targetLabel);
                const pointPrompt = (!activePointSession.logitsToken && !initialMaskRle) ? targetPrompt : undefined;
                const nextResult = await serverProxy.jobs.inferSAM3Points(job.id, {
                    model_id: selectedModelID,
                    session_id: activeSession.sessionID,
                    label_name: targetLabel.name,
                    prompt: pointPrompt,
                    points,
                    logits_token: activePointSession.logitsToken,
                    initial_mask_rle: initialMaskRle,
                    multimask_output: true,
                    score_threshold: scoreThreshold,
                }) as SerializedSAM3PointsResult;

                return {
                    session: activeSession,
                    result: nextResult,
                    localPointBBox: nextLocalPointBBox,
                };
            });

            if (!result.mask_rle) {
                const selectedObjectBBox = activePointSession.originalState ?
                    extractSnapshotBBox(activePointSession.originalState) :
                    null;
                const boxFallbackBBox = localPointBBox || selectedObjectBBox;

                if (boxFallbackBBox) {
                    const boxFallback = await withSAM3SessionRetry(async (activeSession) => (
                        serverProxy.jobs.inferSAM3Box(job.id, {
                            model_id: selectedModelID,
                            session_id: activeSession.sessionID,
                            label_name: targetLabel.name,
                            prompt: targetPrompt,
                            bbox: boxFallbackBBox,
                            negative: false,
                            score_threshold: scoreThreshold,
                        }) as SerializedSAM3MaskResult
                    ));

                    if (boxFallback.mask_rle) {
                        const fallbackTargetClientID = activePointSession.targetClientID ?? activePointSession.sourceClientID;
                        if (fallbackTargetClientID === null) {
                            const mutation = await putMaskObject(targetLabel, boxFallback.mask_rle, Source.SEMI_AUTO);
                            const nextSession = {
                                ...activePointSession,
                                targetClientID: mutation.primaryClientID,
                                auxiliaryClientIDs: mergeClientIDs(
                                    activePointSession.auxiliaryClientIDs,
                                    mutation.auxiliaryClientIDs,
                                ),
                                createdDuringSession: true,
                                replacedExistingObject: false,
                                logitsToken: null,
                            };
                            pointSessionRef.current = nextSession;
                            setPointSession(nextSession);
                            return;
                        }

                        const targetState = await findObjectState(fallbackTargetClientID);
                        if (!targetState) {
                            throw new Error('Selected object is no longer available');
                        }

                        const mutation = await replaceMaskObjectLabel(
                            targetState,
                            targetLabel,
                            boxFallback.mask_rle,
                            Source.SEMI_AUTO,
                        );
                        const nextSession = {
                            ...activePointSession,
                            targetClientID: mutation.primaryClientID,
                            auxiliaryClientIDs: mergeClientIDs(
                                activePointSession.auxiliaryClientIDs,
                                mutation.auxiliaryClientIDs,
                            ),
                            createdDuringSession: false,
                            replacedExistingObject: true,
                            logitsToken: null,
                        };
                        pointSessionRef.current = nextSession;
                        setPointSession(nextSession);
                        return;
                    }
                }

                if (activePointSession.targetClientID !== null &&
                    activePointSession.originalState?.shapeType === ShapeType.MASK) {
                    const targetState = await findObjectState(activePointSession.targetClientID);
                    if (targetState) {
                        const nextSession = {
                            ...activePointSession,
                            targetClientID: targetState.clientID,
                            logitsToken: activePointSession.logitsToken,
                        };
                        pointSessionRef.current = nextSession;
                        setPointSession(nextSession);
                        return;
                    }
                }

                notification.info({ message: 'SAM3 did not return a mask for the current points' });
                return;
            }

            if (activePointSession.targetClientID === null) {
                if (activePointSession.sourceClientID !== null &&
                    activePointSession.originalState &&
                    activePointSession.originalState.label.id !== targetLabel.id) {
                    const targetState = await findObjectState(activePointSession.sourceClientID);
                    if (!targetState) {
                        throw new Error('Selected object is no longer available');
                    }
                    const mutation = await replaceMaskObjectLabel(
                        targetState,
                        targetLabel,
                        result.mask_rle,
                        Source.SEMI_AUTO,
                    );
                    const nextSession = {
                        ...activePointSession,
                        targetClientID: mutation.primaryClientID,
                        auxiliaryClientIDs: mergeClientIDs(
                            activePointSession.auxiliaryClientIDs,
                            mutation.auxiliaryClientIDs,
                        ),
                        createdDuringSession: false,
                        replacedExistingObject: true,
                        logitsToken: result.logits_token,
                    };
                    pointSessionRef.current = nextSession;
                    setPointSession(nextSession);
                } else {
                    const mutation = await putMaskObject(targetLabel, result.mask_rle, Source.SEMI_AUTO);
                    const nextSession = {
                        ...activePointSession,
                        targetClientID: mutation.primaryClientID,
                        auxiliaryClientIDs: mergeClientIDs(
                            activePointSession.auxiliaryClientIDs,
                            mutation.auxiliaryClientIDs,
                        ),
                        createdDuringSession: true,
                        replacedExistingObject: false,
                        logitsToken: result.logits_token,
                    };
                    pointSessionRef.current = nextSession;
                    setPointSession(nextSession);
                }
            } else {
                const targetState = await findObjectState(activePointSession.targetClientID);
                if (!targetState) {
                    throw new Error('Point-refined object is no longer available');
                }
                const mutation = await updateMaskObject(targetState, result.mask_rle, Source.SEMI_AUTO);
                const nextSession = {
                    ...activePointSession,
                    targetClientID: mutation.primaryClientID,
                    auxiliaryClientIDs: mergeClientIDs(
                        activePointSession.auxiliaryClientIDs,
                        mutation.auxiliaryClientIDs,
                    ),
                    replacedExistingObject: activePointSession.replacedExistingObject || targetState.shapeType !== ShapeType.MASK,
                    logitsToken: result.logits_token,
                };
                pointSessionRef.current = nextSession;
                setPointSession(nextSession);
            }
        } catch (errorData: unknown) {
            notification.error({
                message: 'SAM3 point inference failed',
                description: String(errorData),
            });
        } finally {
            setBusy(false);
        }
    }, [
        ensureSession,
        findObjectState,
        job,
        putMaskObject,
        replaceMaskObjectLabel,
        resolvePromptTarget,
        resolveInitialPointMaskRle,
        selectedMaskObject,
        selectedPromptOverridesSelectedObject,
        selectedModelID,
        updateMaskObject,
        withSAM3SessionRetry,
    ]);

    const beginBoxInteraction = useCallback((negative: boolean) => {
        if (!canvasInstance) return;
        if (!selectedModelID) {
            notification.warning({ message: 'Select a SAM3 model first' });
            return;
        }
        if (negative && !selectedMaskObject) {
            notification.warning({ message: 'Select an object before running negative box refinement' });
            return;
        }
        if (!negative) {
            try {
                resolvePromptTarget();
            } catch (errorData: unknown) {
                notification.warning({ message: String(errorData) });
                return;
            }
        }

        setInteractionMode(negative ? 'box-negative' : 'box-positive');
        ignoreCanvasCancelRef.current = true;
        canvasInstance.cancel();
        canvasInstance.interact({
            enabled: true,
            command: 'draw_box',
            settings: {
                crosshair: true,
            },
        });
    }, [canvasInstance, resolvePromptTarget, selectedMaskObject, selectedModelID]);

    const beginPointInteraction = useCallback((mode: PointMode) => {
        if (!canvasInstance) return;
        if (!selectedModelID) {
            notification.warning({ message: 'Select a SAM3 model first' });
            return;
        }
        try {
            resolvePromptTarget({
                preferSelectedPrompt: mode === 1 && selectedPromptOverridesSelectedObject,
            });
        } catch (errorData: unknown) {
            notification.warning({ message: String(errorData) });
            return;
        }

        const existingSession = pointSessionRef.current;
        const nextSession: PointSessionState = existingSession || {
            targetClientID: selectedPromptOverridesSelectedObject ? null : (selectedMaskObject?.clientID ?? null),
            sourceClientID: selectedMaskObject?.clientID ?? null,
            auxiliaryClientIDs: [],
            createdDuringSession: false,
            replacedExistingObject: false,
            originalState: selectedMaskObject ? snapshotShapeState(selectedMaskObject) : null,
            logitsToken: null,
            mode,
        };
        nextSession.mode = mode;
        pointSessionRef.current = nextSession;
        setPointSession({ ...nextSession });
        setInteractionMode('points');
        ignoreCanvasCancelRef.current = true;
        canvasInstance.cancel();
        canvasInstance.interact({
            enabled: true,
            command: 'draw_points',
            settings: {
                appendCursorPositionAsPoint: false,
                removalStrategy: 'any',
                points_type: mode === 1 ? 'positive' : 'negative',
                crosshair: false,
            },
        });
    }, [canvasInstance, resolvePromptTarget, selectedMaskObject, selectedModelID, selectedPromptOverridesSelectedObject]);

    useEffect(() => {
        if (!canvasInstance) return undefined;

        const interactionListener = (event: Event): void => {
            if (busyRef.current) return;
            const { shapes, finished } = (event as CustomEvent<{
                shapes: InteractionResult[];
                finished: boolean;
            }>).detail;

            if (interactionMode === 'box-positive' || interactionMode === 'box-negative') {
                if (!finished || !shapes.length) {
                    return;
                }
                const bbox = normalizeBBox(shapes[0].points);
                endCanvasInteraction();
                void runBoxInference(interactionMode === 'box-negative', bbox);
                return;
            }

            if (interactionMode === 'points') {
                const positivePoints = convertShapesForInteractor(shapes, 'points', 'positive')
                    .map(([x, y]) => ({ x, y, label: 1 }));
                const negativePoints = convertShapesForInteractor(shapes, 'points', 'negative')
                    .map(([x, y]) => ({ x, y, label: 0 }));
                const points = [...positivePoints, ...negativePoints];
                if (!points.length) {
                    if (finished) {
                        void finishPointSession(false);
                    }
                    return;
                }
                void runPointInference(points).then(() => {
                    if (finished) {
                        void finishPointSession(false);
                    }
                });
            }
        };

        const cancelListener = (): void => {
            if (ignoreCanvasCancelRef.current) {
                ignoreCanvasCancelRef.current = false;
                return;
            }
            if (interactionMode === 'points') {
                void finishPointSession(true);
                return;
            }
            setInteractionMode(null);
        };

        canvasInstance.html().addEventListener('canvas.interacted', interactionListener);
        canvasInstance.html().addEventListener('canvas.canceled', cancelListener);

        return () => {
            canvasInstance.html().removeEventListener('canvas.interacted', interactionListener);
            canvasInstance.html().removeEventListener('canvas.canceled', cancelListener);
        };
    }, [
        canvasInstance,
        endCanvasInteraction,
        finishPointSession,
        interactionMode,
        runBoxInference,
        runPointInference,
    ]);

    // ── Video tracking ───────────────────────────────────────────────

    const jobStartFrame = job?.startFrame ?? 0;
    const jobStopFrame = job?.stopFrame ?? 0;

    useEffect(() => {
        if (job) {
            setVideoFrameRange([jobStartFrame, Math.min(jobStartFrame + 4, jobStopFrame)]);
        }
    }, [job, jobStartFrame, jobStopFrame]);

    const startVideoSession = useCallback(async () => {
        if (!job || selectedModelID === null || busy) return;
        setBusy(true);
        try {
            const response: SerializedSAM3VideoSession = await serverProxy.jobs.createSAM3VideoSession(
                job.id,
                {
                    model_id: selectedModelID,
                    start_frame: videoFrameRange[0],
                    stop_frame: videoFrameRange[1],
                },
            );
            setVideoTracking({
                sessionID: response.session_id,
                modelID: selectedModelID,
                startFrame: videoFrameRange[0],
                stopFrame: videoFrameRange[1],
                numFrames: response.num_frames,
                width: response.width,
                height: response.height,
                hasPrompt: false,
                objectLabelIDs: {},
                seededSourceClientIDsByFrame: {},
                seededObjectIDsByFrame: {},
                generatedClientIDsByFrame: {},
                generatedObjectIDsByFrame: {},
                propagating: false,
                progress: 0,
                results: [],
            });
            notification.success({ message: `Video session created (${response.num_frames} frames)` });
        } catch (err: any) {
            notification.error({ message: 'Failed to create video session', description: err?.message });
        } finally {
            setBusy(false);
        }
    }, [job, selectedModelID, videoFrameRange, busy]);

    const applyVideoResultsToAnnotations = useCallback(async (
        results: SerializedSAM3PropagationResult[],
        objectLabelIDs: Record<number, number>,
        generatedClientIDsByFrame: Record<number, number[]>,
        generatedObjectIDsByFrame: Record<number, Record<number, number>>,
    ): Promise<VideoAnnotationApplyResult> => {
        if (!job) {
            throw new Error('No open job');
        }

        const nextGeneratedClientIDsByFrame: Record<number, number[]> = { ...generatedClientIDsByFrame };
        const nextGeneratedObjectIDsByFrame: Record<number, Record<number, number>> = { ...generatedObjectIDsByFrame };
        const masksByFrame = new Map<number, SerializedSAM3VideoMask[]>();
        for (const result of results) {
            const existingMasks = masksByFrame.get(result.frame_index) || [];
            existingMasks.push(...(result.masks || []));
            masksByFrame.set(result.frame_index, existingMasks);
        }

        for (const [frameIndex, frameMasks] of masksByFrame.entries()) {
            const frameStates = await job.annotations.get(frameIndex, false, []);
            const existingClientIDs = nextGeneratedClientIDsByFrame[frameIndex] || [];
            for (const clientID of existingClientIDs) {
                const existingState = frameStates.find((state) => state.clientID === clientID);
                if (existingState) {
                    await existingState.delete(frameIndex, false);
                }
            }

            const masksToCreate = frameMasks.flatMap((mask) => {
                const labelID = objectLabelIDs[mask.obj_id];
                const label = eligibleLabelsByID.get(labelID);
                if (!label || !Array.isArray(mask.mask_rle) || !mask.mask_rle.length) {
                    return [];
                }

                return [{
                    objID: mask.obj_id,
                    state: buildMaskState(
                        label,
                        frameIndex,
                        curZOrder,
                        mask.mask_rle,
                        Source.SEMI_AUTO,
                    ),
                }];
            });

            if (masksToCreate.length) {
                const createdClientIDs = await job.annotations.put(
                    masksToCreate.map((entry) => entry.state),
                );
                nextGeneratedClientIDsByFrame[frameIndex] = createdClientIDs;
                nextGeneratedObjectIDsByFrame[frameIndex] = Object.fromEntries(
                    createdClientIDs.map((clientID, index) => [clientID, masksToCreate[index].objID]),
                );
            } else {
                delete nextGeneratedClientIDsByFrame[frameIndex];
                delete nextGeneratedObjectIDsByFrame[frameIndex];
            }
        }

        await refreshAnnotations();
        return {
            generatedClientIDsByFrame: nextGeneratedClientIDsByFrame,
            generatedObjectIDsByFrame: nextGeneratedObjectIDsByFrame,
        };
    }, [curZOrder, eligibleLabelsByID, job, refreshAnnotations]);

    const addVideoPrompt = useCallback(async () => {
        if (!job || !videoTracking || busy) return;
        if (frame < videoTracking.startFrame || frame > videoTracking.stopFrame) {
            notification.error({
                message: 'Frame outside video session',
                description: `Move to a frame between ${videoTracking.startFrame} and ${videoTracking.stopFrame} before seeding prompts.`,
            });
            return;
        }

        if (!currentFrameVideoPromptObjects.length) {
            notification.warning({
                message: 'No masks to seed',
                description: 'The current frame has no eligible masks to use as video prompts.',
            });
            return;
        }

        setBusy(true);
        try {
            const nextObjectLabelIDs = { ...videoTracking.objectLabelIDs };
            const nextSeededSourceClientIDsByFrame = {
                ...videoTracking.seededSourceClientIDsByFrame,
                [frame]: [...(videoTracking.seededSourceClientIDsByFrame[frame] || [])],
            };
            const nextSeededObjectIDsByFrame = {
                ...videoTracking.seededObjectIDsByFrame,
                [frame]: { ...(videoTracking.seededObjectIDsByFrame[frame] || {}) },
            };
            const seededResults: SerializedSAM3PropagationResult[] = [];
            const skippedLabels: string[] = [];
            const failedLabels: string[] = [];

            for (const state of currentFrameVideoPromptObjects) {
                const label = state.label as Label;
                const normalizedBBox = encodeBBoxToNormalizedXYWH(
                    extractStateBBox(state),
                    videoTracking.width,
                    videoTracking.height,
                );
                if (!normalizedBBox) {
                    skippedLabels.push(label.name);
                    continue;
                }

                try {
                    const result: SerializedSAM3PropagationResult = await serverProxy.jobs.addSAM3VideoPrompt(
                        job.id,
                        {
                            model_id: videoTracking.modelID,
                            session_id: videoTracking.sessionID,
                            frame,
                            bounding_boxes: [normalizedBBox],
                            bounding_box_labels: [1],
                        },
                    );
                    const maskCount = result.masks?.length ?? 0;
                    if (!maskCount) {
                        skippedLabels.push(label.name);
                        continue;
                    }

                    for (const mask of result.masks || []) {
                        nextObjectLabelIDs[mask.obj_id] = label.id;
                    }
                    if ((result.masks?.length || 0) === 1) {
                        nextSeededObjectIDsByFrame[frame][state.clientID] = result.masks[0].obj_id;
                    }
                    seededResults.push(result);
                    nextSeededSourceClientIDsByFrame[frame].push(state.clientID);
                } catch {
                    failedLabels.push(label.name);
                }
            }

            if (!seededResults.length) {
                notification.warning({
                    message: `No prompts were seeded on frame ${frame}`,
                    description: 'None of the current-frame masks produced a usable SAM3 video prompt.',
                });
                return;
            }

            const applyResult = await applyVideoResultsToAnnotations(
                seededResults,
                nextObjectLabelIDs,
                videoTracking.generatedClientIDsByFrame,
                videoTracking.generatedObjectIDsByFrame,
            );
            const seededMaskCount = seededResults.reduce(
                (total, result) => total + (result.masks?.length ?? 0),
                0,
            );
            const summaryParts = [
                `${seededMaskCount} masks seeded`,
                skippedLabels.length ? `${skippedLabels.length} skipped` : null,
                failedLabels.length ? `${failedLabels.length} failed` : null,
            ].filter(Boolean).join(' • ');
            setVideoTracking((prev) => prev && ({
                ...prev,
                hasPrompt: true,
                objectLabelIDs: nextObjectLabelIDs,
                seededSourceClientIDsByFrame: nextSeededSourceClientIDsByFrame,
                seededObjectIDsByFrame: nextSeededObjectIDsByFrame,
                generatedClientIDsByFrame: applyResult.generatedClientIDsByFrame,
                generatedObjectIDsByFrame: applyResult.generatedObjectIDsByFrame,
                results: seededResults,
            }));
            notification.success({
                message: `Seeded current-frame masks on frame ${frame}`,
                description: summaryParts,
            });
        } catch (err: any) {
            notification.error({ message: 'Failed to add prompt', description: err?.message });
        } finally {
            setBusy(false);
        }
    }, [
        applyVideoResultsToAnnotations,
        busy,
        currentFrameVideoPromptObjects,
        frame,
        job,
        videoTracking,
    ]);

    const seedVideoTextPrompts = useCallback(async () => {
        if (!job || !videoTracking || busy) return;
        if (frame < videoTracking.startFrame || frame > videoTracking.stopFrame) {
            notification.error({
                message: 'Frame outside video session',
                description: `Move to a frame between ${videoTracking.startFrame} and ${videoTracking.stopFrame} before seeding prompts.`,
            });
            return;
        }

        const executableBindings = promptBindings.filter((binding) => (
            textPromptKeys.includes(binding.option.key) && binding.label
        ));
        if (!executableBindings.length) {
            notification.warning({
                message: 'No mapped anatomy prompts selected',
                description: 'Select at least one mapped anatomy prompt before seeding video text prompts.',
            });
            return;
        }

        setBusy(true);
        try {
            const nextObjectLabelIDs = { ...videoTracking.objectLabelIDs };
            const seededResults: SerializedSAM3PropagationResult[] = [];
            const skippedPrompts: string[] = [];
            const failedPrompts: string[] = [];

            for (const binding of executableBindings) {
                try {
                    const result: SerializedSAM3PropagationResult = await serverProxy.jobs.addSAM3VideoPrompt(
                        job.id,
                        {
                            model_id: videoTracking.modelID,
                            session_id: videoTracking.sessionID,
                            frame,
                            text: binding.option.prompt,
                        },
                    );
                    const maskCount = result.masks?.length ?? 0;
                    if (!maskCount) {
                        skippedPrompts.push(binding.option.display_name);
                        continue;
                    }

                    for (const mask of result.masks || []) {
                        nextObjectLabelIDs[mask.obj_id] = (binding.label as Label).id;
                    }
                    seededResults.push(result);
                } catch {
                    failedPrompts.push(binding.option.display_name);
                }
            }

            if (!seededResults.length) {
                notification.warning({
                    message: `No anatomy prompts produced masks on frame ${frame}`,
                    description: 'The selected text prompts did not return any usable video masks on this frame.',
                });
                return;
            }

            const applyResult = await applyVideoResultsToAnnotations(
                seededResults,
                nextObjectLabelIDs,
                videoTracking.generatedClientIDsByFrame,
                videoTracking.generatedObjectIDsByFrame,
            );
            const seededMaskCount = seededResults.reduce(
                (total, result) => total + (result.masks?.length ?? 0),
                0,
            );
            const summaryParts = [
                `${seededMaskCount} masks seeded`,
                skippedPrompts.length ? `${skippedPrompts.length} skipped` : null,
                failedPrompts.length ? `${failedPrompts.length} failed` : null,
            ].filter(Boolean).join(' • ');

            setVideoTracking((prev) => prev && ({
                ...prev,
                hasPrompt: true,
                objectLabelIDs: nextObjectLabelIDs,
                generatedClientIDsByFrame: applyResult.generatedClientIDsByFrame,
                generatedObjectIDsByFrame: applyResult.generatedObjectIDsByFrame,
                results: seededResults,
            }));
            notification.success({
                message: `Seeded anatomy prompts on frame ${frame}`,
                description: summaryParts,
            });
        } catch (err: any) {
            notification.error({ message: 'Failed to seed anatomy prompts', description: err?.message });
        } finally {
            setBusy(false);
        }
    }, [
        applyVideoResultsToAnnotations,
        busy,
        frame,
        job,
        promptBindings,
        textPromptKeys,
        videoTracking,
    ]);

    const syncSelectedMaskToVideo = useCallback(async () => {
        if (!job || !videoTracking || busy || !selectedMaskObject) return;
        if (frame < videoTracking.startFrame || frame > videoTracking.stopFrame) {
            notification.error({
                message: 'Frame outside video session',
                description: `Move to a frame between ${videoTracking.startFrame} and ${videoTracking.stopFrame} before syncing corrections.`,
            });
            return;
        }

        const normalizedBBox = encodeBBoxToNormalizedXYWH(
            extractStateBBox(selectedMaskObject),
            videoTracking.width,
            videoTracking.height,
        );
        if (!normalizedBBox) {
            notification.warning({
                message: 'Selected mask cannot be synced',
                description: 'The selected object does not have a valid bounding box for video prompting.',
            });
            return;
        }

        setBusy(true);
        try {
            const label = selectedMaskObject.label as Label;
            const nextObjectLabelIDs = { ...videoTracking.objectLabelIDs };
            const nextSeededSourceClientIDsByFrame = {
                ...videoTracking.seededSourceClientIDsByFrame,
                [frame]: [...(videoTracking.seededSourceClientIDsByFrame[frame] || [])],
            };
            const nextSeededObjectIDsByFrame = {
                ...videoTracking.seededObjectIDsByFrame,
                [frame]: { ...(videoTracking.seededObjectIDsByFrame[frame] || {}) },
            };

            const result: SerializedSAM3PropagationResult = await serverProxy.jobs.addSAM3VideoPrompt(
                job.id,
                {
                    model_id: videoTracking.modelID,
                    session_id: videoTracking.sessionID,
                    frame,
                    bounding_boxes: [normalizedBBox],
                    bounding_box_labels: [1],
                    obj_id: selectedVideoObjectID ?? undefined,
                },
            );
            const maskCount = result.masks?.length ?? 0;
            if (!maskCount) {
                notification.warning({
                    message: `Selected mask did not update video tracking on frame ${frame}`,
                    description: 'SAM3 did not return any masks for the selected object on this frame.',
                });
                return;
            }

            for (const mask of result.masks || []) {
                nextObjectLabelIDs[mask.obj_id] = label.id;
            }
            if ((result.masks?.length || 0) === 1) {
                nextSeededObjectIDsByFrame[frame][selectedMaskObject.clientID] = result.masks[0].obj_id;
            }
            if (selectedVideoObjectID === null &&
                !nextSeededSourceClientIDsByFrame[frame].includes(selectedMaskObject.clientID)) {
                nextSeededSourceClientIDsByFrame[frame].push(selectedMaskObject.clientID);
            }

            const applyResult = await applyVideoResultsToAnnotations(
                [result],
                nextObjectLabelIDs,
                videoTracking.generatedClientIDsByFrame,
                videoTracking.generatedObjectIDsByFrame,
            );

            setVideoTracking((prev) => prev && ({
                ...prev,
                hasPrompt: true,
                objectLabelIDs: nextObjectLabelIDs,
                seededSourceClientIDsByFrame: nextSeededSourceClientIDsByFrame,
                seededObjectIDsByFrame: nextSeededObjectIDsByFrame,
                generatedClientIDsByFrame: applyResult.generatedClientIDsByFrame,
                generatedObjectIDsByFrame: applyResult.generatedObjectIDsByFrame,
                results: [result],
            }));
            notification.success({
                message: selectedVideoObjectID === null ?
                    `Seeded selected mask on frame ${frame}` :
                    `Synced selected mask correction on frame ${frame}`,
                description: `${maskCount} masks updated`,
            });
        } catch (err: any) {
            notification.error({ message: 'Failed to sync selected mask', description: err?.message });
        } finally {
            setBusy(false);
        }
    }, [
        applyVideoResultsToAnnotations,
        busy,
        frame,
        job,
        selectedMaskObject,
        selectedVideoObjectID,
        videoTracking,
    ]);

    const runVideoPropagation = useCallback(async (direction: string) => {
        if (!job || !videoTracking || videoTracking.propagating) return;
        if (!videoTracking.hasPrompt) {
            notification.error({
                message: 'Propagation needs a prompt',
                description: 'Add a video prompt on a frame before running propagation.',
            });
            return;
        }
        setVideoTracking((prev) => prev && ({ ...prev, propagating: true, progress: 0, results: [] }));
        try {
            const allResults = await serverProxy.jobs.propagateSAM3Video(
                job.id,
                {
                    model_id: videoTracking.modelID,
                    session_id: videoTracking.sessionID,
                    direction,
                },
                (result) => {
                    setVideoTracking((prev) => {
                        if (!prev) return prev;
                        const newResults = [...prev.results, result];
                        return {
                            ...prev,
                            results: newResults,
                            progress: Math.round((newResults.length / prev.numFrames) * 100),
                        };
                    });
                },
            );
            const applyResult = await applyVideoResultsToAnnotations(
                allResults,
                videoTracking.objectLabelIDs,
                videoTracking.generatedClientIDsByFrame,
                videoTracking.generatedObjectIDsByFrame,
            );
            setVideoTracking((prev) => prev && ({
                ...prev,
                propagating: false,
                progress: 100,
                generatedClientIDsByFrame: applyResult.generatedClientIDsByFrame,
                generatedObjectIDsByFrame: applyResult.generatedObjectIDsByFrame,
                results: allResults,
            }));
            notification.success({
                message: 'Propagation complete',
                description: `${allResults.length} frames processed`,
            });
        } catch (err: any) {
            notification.error({ message: 'Propagation failed', description: err?.message });
            setVideoTracking((prev) => prev && ({ ...prev, propagating: false }));
        }
    }, [applyVideoResultsToAnnotations, job, videoTracking]);

    const closeVideoSession = useCallback(async () => {
        if (!job || !videoTracking) return;
        try {
            await serverProxy.jobs.deleteSAM3VideoSession(
                job.id, videoTracking.sessionID, videoTracking.modelID,
            );
        } catch {
            // ignore cleanup errors
        }
        setVideoTracking(null);
    }, [job, videoTracking]);

    if (!job) {
        return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description='No active job' />;
    }

    return (
        <div className='cvat-surgery-sam3-panel'>
            <div className='cvat-surgery-sam3-header'>
                <div>
                    <Text strong>SAM3 Anatomy</Text>
                    <div className='cvat-surgery-sam3-subtitle'>
                        Interactive anatomy masks with procedure-specific variants
                    </div>
                </div>
                <Button
                    icon={<ReloadOutlined />}
                    onClick={() => void loadModels()}
                    loading={modelsLoading}
                />
            </div>

            {modelsLoading ? (
                <div className='cvat-surgery-sam3-loading'>
                    <Spin />
                </div>
            ) : !models.length ? (
                <Alert
                    type='info'
                    showIcon
                    message='No SAM3 models'
                    description='No anatomy_segmenter models are registered for this job procedure type.'
                />
            ) : (
                <>
                    <div className='cvat-surgery-sam3-section'>
                        <Text className='cvat-text-color'>Model variant</Text>
                        <Select
                            className='cvat-surgery-sam3-model-select'
                            value={selectedModelID ?? undefined}
                            onChange={(value: number) => setSelectedModelID(value)}
                            options={modelOptions}
                        />
                        {selectedModel && (
                            <div className='cvat-surgery-sam3-model-meta'>
                                <Text type='secondary'>
                                    {`Device: ${selectedModel.device} • Priority: ${selectedModel.priority}${selectedModel.is_default ? ' • Default' : ''}${selectedModel.is_matching_procedure ? '' : ' • Fallback model'}`}
                                </Text>
                            </div>
                        )}
                    </div>

                    <Divider />

                    <div className='cvat-surgery-sam3-section'>
                        <Text className='cvat-text-color'>Thresholds</Text>
                        <div className='cvat-surgery-sam3-threshold-row'>
                            <Text type='secondary'>{`Confidence threshold: ${confidenceThreshold.toFixed(2)}`}</Text>
                            <Slider
                                min={0}
                                max={1}
                                step={0.05}
                                value={confidenceThreshold}
                                onChange={(value) => setConfidenceThreshold(Number(value))}
                            />
                        </div>
                        <div className='cvat-surgery-sam3-threshold-row'>
                            <Text type='secondary'>{`Score threshold: ${scoreThreshold.toFixed(2)}`}</Text>
                            <Slider
                                min={0}
                                max={1}
                                step={0.05}
                                value={scoreThreshold}
                                onChange={(value) => setScoreThreshold(Number(value))}
                            />
                        </div>
                    </div>

                    <Divider />

                    <div className='cvat-surgery-sam3-section'>
                        <Text className='cvat-text-color'>Anatomy prompts</Text>
                        <Text type='secondary'>
                            Use the selected model&apos;s trained anatomy prompt vocabulary instead of project phase labels.
                        </Text>
                        <Select
                            className='cvat-surgery-sam3-label-select'
                            value={selectedPromptKey ?? undefined}
                            onChange={(value: string) => setSelectedPromptKey(value)}
                            options={promptOptions}
                            placeholder='Select an anatomy prompt'
                        />
                        <div className='cvat-surgery-sam3-context'>
                            <Text type='secondary'>
                                {selectedMaskObject ?
                                    `Selected object: ${selectedMaskObject.label.name}` :
                                    `Working prompt: ${selectedPromptBinding?.option.display_name || activePromptBinding?.option.display_name || 'None'}`}
                            </Text>
                        </div>
                        <div className='cvat-surgery-sam3-context'>
                            <Text type='secondary'>
                                {selectedMaskObject ?
                                    `Prompt sent to SAM3: ${selectedObjectPromptBinding?.option.prompt || selectedMaskObject.label.name}` :
                                    `Target CVAT label: ${selectedPromptBinding?.label?.name || activePromptBinding?.label?.name || 'No matching anatomy label'}`}
                            </Text>
                        </div>
                        {!!unmappedPromptBindings.length && (
                            <Alert
                                type='warning'
                                showIcon
                                message='Some anatomy prompts do not have matching CVAT labels'
                                description={(
                                    <div className='cvat-surgery-sam3-alert-body'>
                                        {`Masks can only be created for prompts that match an existing CVAT mask/any label. Missing mappings: ${unmappedPromptBindings.map((binding) => binding.option.display_name).join(', ')}`}
                                    </div>
                                )}
                            />
                        )}
                        {!!unmappedPromptBindings.length && (
                            <Button
                                onClick={() => void syncPromptLabels()}
                                disabled={!selectedModelID || busy}
                            >
                                Create missing anatomy labels
                            </Button>
                        )}
                    </div>

                    <Divider />

                    <div className='cvat-surgery-sam3-section'>
                        <Text className='cvat-text-color'>Interactive refinement</Text>
                        <Space wrap>
                            <Button
                                type='primary'
                                icon={<ScissorOutlined />}
                                onClick={() => void runFullImageAutoInference()}
                                disabled={!selectedModelID || busy}
                            >
                                Full image auto
                            </Button>
                            <Button
                                icon={<ScissorOutlined />}
                                onClick={() => void runFullImageInference()}
                                disabled={!selectedModelID || busy}
                            >
                                Full image prompt
                            </Button>
                            <Button
                                icon={<PlusOutlined />}
                                onClick={() => beginBoxInteraction(false)}
                                disabled={!selectedModelID || busy}
                            >
                                Positive box
                            </Button>
                            <Button
                                onClick={() => beginBoxInteraction(true)}
                                disabled={!selectedModelID || !selectedMaskObject || busy}
                            >
                                Negative box
                            </Button>
                            <Button
                                type={interactionMode === 'points' && pointSession?.mode === 1 ? 'primary' : 'default'}
                                onClick={() => beginPointInteraction(1)}
                                disabled={!selectedModelID || busy}
                            >
                                Positive points
                            </Button>
                            <Button
                                type={interactionMode === 'points' && pointSession?.mode === 0 ? 'primary' : 'default'}
                                onClick={() => beginPointInteraction(0)}
                                disabled={!selectedModelID || busy}
                            >
                                Negative points
                            </Button>
                        </Space>
                        {!hasPromptTarget && (
                            <Text type='secondary'>
                                Select a mapped anatomy prompt or create the missing anatomy labels to enable new mask creation.
                            </Text>
                        )}
                    </div>

                    <Divider />

                    <div className='cvat-surgery-sam3-section'>
                        <Text className='cvat-text-color'>Text prelabel prompts</Text>
                        <Text type='secondary'>
                            {mappedPromptBindings.length ?
                                `Selected prompts drive both text prelabel and full-image auto inference. Mapped prompts: ${mappedPromptBindings.length}` :
                                'No mapped anatomy prompts are currently available for text prelabel.'}
                        </Text>
                        <Select
                            mode='multiple'
                            className='cvat-surgery-sam3-label-select'
                            value={textPromptKeys}
                            onChange={(value: string[]) => setTextPromptKeys(value)}
                            options={textPromptOptions}
                            maxTagCount='responsive'
                            placeholder='Select prompts for text prelabel'
                        />
                        <Button
                            type='primary'
                            icon={<ScissorOutlined />}
                            onClick={() => void runTextPrelabel()}
                            loading={busy}
                            disabled={!selectedModelID || !textPromptKeys.length}
                        >
                            Text prelabel
                        </Button>
                    </div>

                    {pointSession && (
                        <>
                            <Divider />
                            <Alert
                                type='info'
                                showIcon
                                message='Point refinement active'
                                description={(
                                    <div className='cvat-surgery-sam3-point-session'>
                                        <div>
                                            Click to refine the current object. Use Finish to keep the latest mask, or Cancel to revert.
                                        </div>
                                        <Space wrap>
                                            <Button
                                                type='primary'
                                                icon={<CheckOutlined />}
                                                onClick={() => void finishPointSession(false)}
                                                disabled={busy}
                                            >
                                                Finish points
                                            </Button>
                                            <Button
                                                icon={<StopOutlined />}
                                                onClick={() => void finishPointSession(true)}
                                                disabled={busy}
                                            >
                                                Cancel points
                                            </Button>
                                        </Space>
                                    </div>
                                )}
                            />
                        </>
                    )}

                    <Divider />

                    <Collapse
                        ghost
                        items={[{
                            key: 'video-tracking',
                            label: (
                                <Text strong>
                                    <FastForwardOutlined />
                                    {' '}
                                    Video Tracking (SAM 3.1)
                                </Text>
                            ),
                            children: (
                                <div className='cvat-surgery-sam3-section'>
                                    {!videoTracking ? (
                                        <>
                                            <Text type='secondary'>
                                                Track objects across multiple frames using SAM 3.1 Object Multiplex.
                                            </Text>
                                            <div className='cvat-surgery-sam3-threshold-row'>
                                                <Text type='secondary'>Frame range:</Text>
                                                <Space>
                                                    <InputNumber
                                                        size='small'
                                                        min={jobStartFrame}
                                                        max={jobStopFrame}
                                                        value={videoFrameRange[0]}
                                                        onChange={(v) => v !== null && setVideoFrameRange(
                                                            [v, Math.max(v, videoFrameRange[1])],
                                                        )}
                                                    />
                                                    <Text type='secondary'>to</Text>
                                                    <InputNumber
                                                        size='small'
                                                        min={videoFrameRange[0]}
                                                        max={jobStopFrame}
                                                        value={videoFrameRange[1]}
                                                        onChange={(v) => v !== null && setVideoFrameRange(
                                                            [videoFrameRange[0], v],
                                                        )}
                                                    />
                                                </Space>
                                            </div>
                                            <Button
                                                type='primary'
                                                icon={<FastForwardOutlined />}
                                                onClick={() => void startVideoSession()}
                                                loading={busy}
                                                disabled={!selectedModelID}
                                            >
                                                Start video session
                                            </Button>
                                        </>
                                    ) : (
                                        <>
                                            <Alert
                                                type='success'
                                                showIcon
                                                message={`Video session active (${videoTracking.numFrames} frames)`}
                                                description={`Frames ${videoTracking.startFrame}–${videoTracking.stopFrame}`}
                                            />
                                            <Space direction='vertical' style={{ width: '100%', marginTop: 8 }}>
                                                <Button
                                                    onClick={() => void seedVideoTextPrompts()}
                                                    disabled={
                                                        busy ||
                                                        videoTracking.propagating ||
                                                        frame < videoTracking.startFrame ||
                                                        frame > videoTracking.stopFrame ||
                                                        !textPromptKeys.length
                                                    }
                                                >
                                                    Seed selected prompts on current frame
                                                </Button>
                                                <Button
                                                    onClick={() => void addVideoPrompt()}
                                                    disabled={
                                                        busy ||
                                                        videoTracking.propagating ||
                                                        frame < videoTracking.startFrame ||
                                                        frame > videoTracking.stopFrame ||
                                                        !currentFrameVideoPromptObjects.length
                                                    }
                                                >
                                                    Seed all masks on current frame
                                                </Button>
                                                <Button
                                                    onClick={() => void syncSelectedMaskToVideo()}
                                                    disabled={
                                                        busy ||
                                                        videoTracking.propagating ||
                                                        frame < videoTracking.startFrame ||
                                                        frame > videoTracking.stopFrame ||
                                                        !selectedMaskObject
                                                    }
                                                >
                                                    Sync selected mask to video
                                                </Button>
                                                <Space wrap>
                                                    <Button
                                                        type='primary'
                                                        icon={<FastForwardOutlined />}
                                                        onClick={() => void runVideoPropagation('forward')}
                                                        loading={videoTracking.propagating}
                                                        disabled={!videoTracking.hasPrompt}
                                                    >
                                                        Forward
                                                    </Button>
                                                    <Button
                                                        onClick={() => void runVideoPropagation('backward')}
                                                        loading={videoTracking.propagating}
                                                        disabled={!videoTracking.hasPrompt}
                                                    >
                                                        Backward
                                                    </Button>
                                                    <Button
                                                        onClick={() => void runVideoPropagation('both')}
                                                        loading={videoTracking.propagating}
                                                        disabled={!videoTracking.hasPrompt}
                                                    >
                                                        Both
                                                    </Button>
                                                </Space>
                                                {!videoTracking.hasPrompt && (
                                                    <Text type='secondary'>
                                                        {frame < videoTracking.startFrame || frame > videoTracking.stopFrame ?
                                                            `Move to a frame between ${videoTracking.startFrame} and ${videoTracking.stopFrame} to seed prompts.` :
                                                            textPromptKeys.length || currentFrameVideoPromptObjects.length || selectedMaskObject ?
                                                                'Seed anatomy prompts, seed current-frame masks, or sync a corrected selected mask before propagation.' :
                                                                'The current frame has no eligible prompts or masks available to seed.'}
                                                    </Text>
                                                )}
                                                {videoTracking.propagating && (
                                                    <Progress percent={videoTracking.progress} size='small' />
                                                )}
                                                {videoTracking.results.length > 0 && !videoTracking.propagating && (
                                                    <Text type='secondary'>
                                                        {`${videoTracking.results.length} frames propagated`}
                                                    </Text>
                                                )}
                                                <Button
                                                    danger
                                                    icon={<StopOutlined />}
                                                    onClick={() => void closeVideoSession()}
                                                    disabled={videoTracking.propagating}
                                                >
                                                    Close video session
                                                </Button>
                                            </Space>
                                        </>
                                    )}
                                </div>
                            ),
                        }]}
                    />
                </>
            )}
        </div>
    );
}
