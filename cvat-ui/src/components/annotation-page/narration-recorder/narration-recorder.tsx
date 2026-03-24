import './styles.scss';
import React, {
    useCallback, useEffect, useRef, useState,
} from 'react';
import Alert from 'antd/lib/alert';
import Button from 'antd/lib/button';
import Space from 'antd/lib/space';
import Typography from 'antd/lib/typography';
import notification from 'antd/lib/notification';
import { useSelector } from 'react-redux';

import serverProxy from 'cvat-core/src/server-proxy';
import { CombinedState } from 'reducers';

type RecorderStatus = 'idle' | 'recording' | 'recorded' | 'uploading';

interface Props {
    onUploadSuccess?: () => void;
}

interface RecorderSupportState {
    supported: boolean;
    message: string;
    description: string;
}

function getRecorderSupportState(): RecorderSupportState {
    if (!window.isSecureContext) {
        return {
            supported: false,
            message: 'Microphone access requires HTTPS or localhost',
            description: `This page is loaded from ${window.location.origin}. Browsers block audio capture on plain HTTP for non-localhost origins.`,
        };
    }

    if (!navigator.mediaDevices?.getUserMedia) {
        return {
            supported: false,
            message: 'Audio recording is not available in this browser',
            description: 'The browser does not expose navigator.mediaDevices.getUserMedia().',
        };
    }

    if (typeof MediaRecorder === 'undefined') {
        return {
            supported: false,
            message: 'Audio recording is not available in this browser',
            description: 'The browser does not expose MediaRecorder.',
        };
    }

    return {
        supported: true,
        message: '',
        description: '',
    };
}

export default function NarrationRecorder(props: Props): JSX.Element {
    const { onUploadSuccess } = props;
    const job = useSelector((state: CombinedState) => state.annotation.job.instance);
    const frameNumber = useSelector((state: CombinedState) => state.annotation.player.frame.number);

    const [status, setStatus] = useState<RecorderStatus>('idle');
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [audioURL, setAudioURL] = useState<string | null>(null);

    const chunksRef = useRef<BlobPart[]>([]);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const startFrameRef = useRef<number | null>(null);
    const startWallclockRef = useRef<string | null>(null);
    const recorderSupport = getRecorderSupportState();

    const cleanup = useCallback((): void => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }
        mediaRecorderRef.current = null;

        if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
        }
        streamRef.current = null;

        chunksRef.current = [];
        startFrameRef.current = null;
        startWallclockRef.current = null;
    }, []);

    useEffect(() => () => {
        cleanup();
    }, [cleanup]);

    useEffect(() => () => {
        if (audioURL) {
            URL.revokeObjectURL(audioURL);
        }
    }, [audioURL]);

    const startRecording = useCallback(async (): Promise<void> => {
        if (status === 'recording') {
            return;
        }

        if (!recorderSupport.supported) {
            notification.error({
                message: recorderSupport.message,
                description: recorderSupport.description,
            });
            return;
        }

        cleanup();
        if (audioURL) {
            URL.revokeObjectURL(audioURL);
            setAudioURL(null);
        }
        setAudioBlob(null);

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;

            const recorder = new MediaRecorder(stream);
            mediaRecorderRef.current = recorder;
            chunksRef.current = [];
            startFrameRef.current = frameNumber;
            startWallclockRef.current = new Date().toISOString();

            recorder.ondataavailable = (event: BlobEvent): void => {
                if (event.data.size) {
                    chunksRef.current.push(event.data);
                }
            };

            recorder.onstop = (): void => {
                const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
                setAudioBlob(blob);
                setAudioURL(URL.createObjectURL(blob));
                setStatus('recorded');

                if (streamRef.current) {
                    streamRef.current.getTracks().forEach((track) => track.stop());
                    streamRef.current = null;
                }
            };

            recorder.start();
            setStatus('recording');
        } catch (error: unknown) {
            notification.error({
                message: 'Could not start recording',
                description: error instanceof Error ? error.message : 'Unknown error',
            });
            cleanup();
            setStatus('idle');
        }
    }, [cleanup, audioURL, frameNumber, recorderSupport, status]);

    const stopRecording = useCallback((): void => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }
    }, []);

    const upload = useCallback(async (): Promise<void> => {
        if (!job || !audioBlob || status === 'uploading') {
            return;
        }

        setStatus('uploading');
        try {
            const filename = `narration_job_${job.id}_${Date.now()}.webm`;
            await serverProxy.jobs.createNarration(job.id, {
                file: audioBlob,
                filename,
                metadata: {
                    start_frame: startFrameRef.current,
                    start_wallclock: startWallclockRef.current,
                    mime_type: audioBlob.type,
                },
            });

            notification.success({ message: 'Narration uploaded' });
            setStatus('recorded');
            onUploadSuccess?.();
        } catch (error: unknown) {
            notification.error({
                message: 'Could not upload narration',
                description: error instanceof Error ? error.message : 'Unknown error',
            });
            setStatus('recorded');
        }
    }, [job, audioBlob, status]);

    return (
        <div className='cvat-narration-recorder'>
            <Space direction='vertical' size={12} style={{ width: '100%' }}>
                <Typography.Text type='secondary'>
                    {job ? `Job #${job.id}` : 'Job is not loaded'}
                </Typography.Text>

                {!recorderSupport.supported ? (
                    <Alert
                        className='cvat-narration-recorder-warning'
                        showIcon
                        type='warning'
                        message={recorderSupport.message}
                        description={(
                            <Space direction='vertical' size={4}>
                                <Typography.Text type='secondary'>
                                    {recorderSupport.description}
                                </Typography.Text>
                                <Typography.Text type='secondary'>
                                    Open CVAT through HTTPS, or through a localhost tunnel on your laptop, to enable recording.
                                </Typography.Text>
                            </Space>
                        )}
                    />
                ) : null}

                <Space wrap>
                    <Button
                        type='primary'
                        onClick={startRecording}
                        disabled={!job || !recorderSupport.supported || status === 'recording' || status === 'uploading'}
                    >
                        Record
                    </Button>
                    <Button onClick={stopRecording} disabled={status !== 'recording'}>
                        Stop
                    </Button>
                    <Button onClick={upload} disabled={!job || !audioBlob || status !== 'recorded'}>
                        Upload
                    </Button>
                </Space>

                {audioURL ? (
                    <audio className='cvat-narration-recorder-audio' controls src={audioURL} />
                ) : (
                    <Typography.Text type='secondary'>No recording yet</Typography.Text>
                )}
            </Space>
        </div>
    );
}
