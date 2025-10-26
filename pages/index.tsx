import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection';

type NormalizedKeypoint = {
  x: number;
  y: number;
  score?: number;
};

type PoseLibrary = Record<string, NormalizedKeypoint[]>;
type PoseCandidate = {
  id: string;
  keypoints: NormalizedKeypoint[];
  thumbnail: string;
};

const NOTES = ['do', 're', 'mi', 'fa', 'so', 'la', 'ti'];
const MATCH_THRESHOLD = 0.35;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const detectorRef = useRef<poseDetection.PoseDetector | null>(null);
  const poseRef = useRef<poseDetection.Pose | null>(null);
  const animationFrameRef = useRef<number>();
  const lastPlayedNoteRef = useRef<string>('');
  const lastPlayTimestampRef = useRef<number>(0);

  const [isLoadingDetector, setIsLoadingDetector] = useState(true);
  const [poseLibrary, setPoseLibrary] = useState<PoseLibrary>({});
  const [selectedNote, setSelectedNote] = useState<string>(NOTES[0]);
  const [mode, setMode] = useState<'train' | 'perform'>('train');
  const [currentNote, setCurrentNote] = useState<string>('None');
  const [statusMessage, setStatusMessage] = useState<string>('Initializing pose detector...');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [isCountdownActive, setIsCountdownActive] = useState(false);
  const [countdownValue, setCountdownValue] = useState<number | null>(null);
  const [isCapturingSequence, setIsCapturingSequence] = useState(false);
  const [captureCandidates, setCaptureCandidates] = useState<PoseCandidate[] | null>(null);
  const [selectedCandidateIndex, setSelectedCandidateIndex] = useState<number | null>(null);
  const [captureProgress, setCaptureProgress] = useState(0);

  const adjacentPairs = useMemo(
    () => poseDetection.util.getAdjacentPairs(poseDetection.SupportedModels.MoveNet),
    []
  );

  useEffect(() => {
    const loadDetector = async () => {
      try {
        await tf.ready();
        if (tf.getBackend() !== 'webgl') {
          await tf.setBackend('webgl');
        }

        const detectorConfig: poseDetection.MoveNetModelConfig = {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        };
        const detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          detectorConfig
        );

        detectorRef.current = detector;
        setStatusMessage('Pose detector ready. Step back so your full body is visible before capturing.');
      } catch (error) {
        console.error('Error loading MoveNet detector:', error);
        setErrorMessage('Unable to load the MoveNet detector. Please refresh the page.');
      } finally {
        setIsLoadingDetector(false);
      }
    };

    loadDetector();

    return () => {
      detectorRef.current?.dispose();
      detectorRef.current = null;
    };
  }, []);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (error) {
        console.error('Error accessing the camera:', error);
        setErrorMessage('Unable to access the camera. Please allow camera permissions and refresh.');
      }
    };

    startCamera();

    return () => {
      const tracks = (videoRef.current?.srcObject as MediaStream | null)?.getTracks();
      tracks?.forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    const detectPoses = async () => {
      if (!detectorRef.current || !videoRef.current) {
        animationFrameRef.current = requestAnimationFrame(detectPoses);
        return;
      }

      const video = videoRef.current;
      const poses = await detectorRef.current.estimatePoses(video, {
        maxPoses: 1,
        flipHorizontal: true,
      });

      const pose = poses[0];
      if (pose) {
        poseRef.current = pose;
        drawPose(pose);

        if (mode === 'perform' && Object.keys(poseLibrary).length > 0) {
          const detectedNote = classifyPose(pose, poseLibrary);
          if (detectedNote) {
            setCurrentNote(detectedNote);
            playNote(detectedNote);
            setStatusMessage(`Detected pose for ${detectedNote.toUpperCase()}`);
          } else {
            setCurrentNote('None');
            setStatusMessage('No matching pose detected. Try adjusting your stance.');
          }
        }
      } else {
        clearCanvas();
        poseRef.current = null;
        if (mode === 'perform') {
          setCurrentNote('None');
          setStatusMessage('No person detected. Step into the frame.');
        }
      }

      animationFrameRef.current = requestAnimationFrame(detectPoses);
    };

    animationFrameRef.current = requestAnimationFrame(detectPoses);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [mode, poseLibrary]);

  useEffect(() => {
    if (!isCountdownActive || countdownValue === null) {
      return;
    }

    if (countdownValue > 0) {
      setStatusMessage(`Hold your pose. Capturing in ${countdownValue}...`);
      const timeoutId = setTimeout(() => {
        setCountdownValue((value) => {
          if (value === null) {
            return value;
          }
          return value > 0 ? value - 1 : value;
        });
      }, 1000);

      return () => clearTimeout(timeoutId);
    }

    setIsCountdownActive(false);
    setCountdownValue(null);
    void collectPoseCandidates();
  }, [countdownValue, isCountdownActive]);

  const startPoseCaptureSequence = () => {
    if (isCountdownActive || isCapturingSequence) {
      return;
    }

    if (!videoRef.current || videoRef.current.videoWidth === 0 || videoRef.current.videoHeight === 0) {
      setStatusMessage('Camera feed not ready yet. Wait a moment and try again.');
      return;
    }

    setCaptureCandidates(null);
    setSelectedCandidateIndex(null);
    setCountdownValue(5);
    setIsCountdownActive(true);
    setStatusMessage('Move into position. Capture begins in 5 seconds.');
  };

  const collectPoseCandidates = async () => {
    if (!videoRef.current) {
      setStatusMessage('Camera feed not available. Refresh the page and try again.');
      return;
    }

    const wait = (ms: number) => new Promise<void>((resolve) => {
      setTimeout(resolve, ms);
    });

    setIsCapturingSequence(true);
    setCaptureProgress(0);
    setStatusMessage('Capturing pose samples...');

    const samples: PoseCandidate[] = [];
    const video = videoRef.current;

    for (let index = 0; index < 5; index += 1) {
      if (index === 0) {
        await wait(200);
      } else {
        await wait(1000);
      }

      const pose = poseRef.current;
      if (!pose) {
        continue;
      }

      const normalizedKeypoints = normalizeKeypoints(pose.keypoints);
      if (!normalizedKeypoints) {
        continue;
      }

      const thumbnail = captureFrameThumbnail(video);
      if (!thumbnail) {
        continue;
      }

      samples.push({
        id: `${Date.now()}-${index}`,
        keypoints: normalizedKeypoints,
        thumbnail,
      });
      setCaptureProgress(samples.length);
    }

    setIsCapturingSequence(false);
    setCaptureProgress(0);

    if (samples.length > 0) {
      setCaptureCandidates(samples);
      setSelectedCandidateIndex(0);
      setStatusMessage('Select your favorite snapshot, then click Save Pose.');
    } else {
      setCaptureCandidates(null);
      setStatusMessage('Could not capture a confident pose. Make sure you are fully in frame and try again.');
    }
  };

  const saveSelectedPose = () => {
    if (!captureCandidates || captureCandidates.length === 0) {
      setStatusMessage('Capture a pose sequence before saving.');
      return;
    }

    if (selectedCandidateIndex === null) {
      setStatusMessage('Select one of the snapshots to keep.');
      return;
    }

    const chosenCandidate = captureCandidates[selectedCandidateIndex];
    setPoseLibrary((prev) => ({
      ...prev,
      [selectedNote]: chosenCandidate.keypoints,
    }));
    setCaptureCandidates(null);
    setSelectedCandidateIndex(null);
    setStatusMessage(`Saved pose for ${selectedNote.toUpperCase()}.`);
  };

  const clearPose = (note: string) => {
    setPoseLibrary((prev) => {
      const updated = { ...prev };
      delete updated[note];
      return updated;
    });
    setStatusMessage(`Cleared saved pose for ${note.toUpperCase()}.`);
    if (note === selectedNote) {
      setCaptureCandidates(null);
      setSelectedCandidateIndex(null);
    }
  };

  const playNote = (note: string) => {
    const now = Date.now();
    if (lastPlayedNoteRef.current === note && now - lastPlayTimestampRef.current < 800) {
      return;
    }

    const audio = new Audio(`/notes/${note}.mp3`);
    audio.play().catch((error) => console.error('Error playing audio:', error));

    lastPlayedNoteRef.current = note;
    lastPlayTimestampRef.current = now;
  };

  const drawPose = (pose: poseDetection.Pose) => {
    if (!canvasRef.current || !videoRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const video = videoRef.current;

    if (!context) {
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    context.clearRect(0, 0, canvas.width, canvas.height);

    context.strokeStyle = '#22c55e';
    context.lineWidth = 4;
    context.fillStyle = '#16a34a';

    pose.keypoints.forEach((keypoint) => {
      if ((keypoint.score ?? 0) < 0.3) {
        return;
      }
      context.beginPath();
      context.arc(keypoint.x, keypoint.y, 6, 0, 2 * Math.PI);
      context.fill();
    });

    adjacentPairs.forEach(([i, j]) => {
      const kp1 = pose.keypoints[i];
      const kp2 = pose.keypoints[j];
      if ((kp1.score ?? 0) < 0.3 || (kp2.score ?? 0) < 0.3) {
        return;
      }
      context.beginPath();
      context.moveTo(kp1.x, kp1.y);
      context.lineTo(kp2.x, kp2.y);
      context.stroke();
    });
  };

  const clearCanvas = () => {
    if (!canvasRef.current) {
      return;
    }
    const context = canvasRef.current.getContext('2d');
    if (context) {
      context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  const poseLibraryCount = useMemo(() => Object.keys(poseLibrary).length, [poseLibrary]);

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      <div className="max-w-5xl mx-auto px-6 py-10">
        <h1 className="text-4xl font-bold mb-6 text-center">FingerMusic Pose Trainer</h1>
        <p className="text-center text-slate-300 mb-8">
          Use your camera to map body poses to musical notes and perform hands-free melodies.
        </p>

        <div className="flex flex-col lg:flex-row gap-6">
          <div className="flex-1 space-y-4">
            <div className="relative rounded-xl overflow-hidden shadow-lg">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full rounded-xl"
              />
              <canvas ref={canvasRef} className="absolute inset-0" />
              {isCountdownActive && countdownValue !== null && (
                <div className="absolute inset-0 bg-slate-900/70 flex items-center justify-center">
                  <span className="text-7xl font-bold text-emerald-300">{countdownValue}</span>
                </div>
              )}
              {isCapturingSequence && (
                <div className="absolute inset-0 bg-slate-900/70 flex flex-col items-center justify-center space-y-2">
                  <span className="text-lg font-semibold">Capturing pose samples...</span>
                  <span className="text-sm text-slate-200">Captured {captureProgress} / 5</span>
                </div>
              )}
            </div>
            <div className="bg-slate-800 rounded-xl p-4 space-y-3">
              <div className="flex flex-wrap gap-3 items-center justify-between">
                <div>
                  <span className="text-sm uppercase tracking-wide text-slate-400">Mode</span>
                  <div className="mt-1 flex gap-2">
                    <button
                      className={`px-3 py-2 rounded-lg text-sm font-semibold transition-colors ${
                        mode === 'train' ? 'bg-emerald-500 text-slate-900' : 'bg-slate-700'
                      }`}
                      onClick={() => {
                        setMode('train');
                        setCurrentNote('None');
                        setStatusMessage('Training mode: use the countdown capture to store poses for each note.');
                      }}
                    >
                      1. Train Poses
                    </button>
                    <button
                      className={`px-3 py-2 rounded-lg text-sm font-semibold transition-colors ${
                        mode === 'perform' ? 'bg-emerald-500 text-slate-900' : 'bg-slate-700'
                      } ${poseLibraryCount === 0 ? 'opacity-50 cursor-not-allowed' : ''}`}
                      onClick={() => {
                        if (poseLibraryCount === 0) {
                          setStatusMessage('Capture at least one pose before switching to performance mode.');
                          return;
                        }
                        setMode('perform');
                        setStatusMessage('Performance mode: hold a trained pose to play its note.');
                      }}
                    >
                      2. Perform
                    </button>
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-sm uppercase tracking-wide text-slate-400 block">Current Note</span>
                  <span className="text-2xl font-semibold text-emerald-400">{currentNote}</span>
                </div>
              </div>

              {mode === 'train' && (
                <div className="space-y-3">
                  <div className="flex flex-col sm:flex-row gap-3 sm:items-center">
                    <label className="text-sm font-medium text-slate-300" htmlFor="note-select">
                      Assign pose to note
                    </label>
                    <select
                      id="note-select"
                      value={selectedNote}
                      onChange={(event) => {
                        setSelectedNote(event.target.value);
                        setCaptureCandidates(null);
                        setSelectedCandidateIndex(null);
                      }}
                      className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-400"
                    >
                      {NOTES.map((note) => (
                        <option key={note} value={note}>
                          {note.toUpperCase()}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="flex flex-wrap gap-3">
                    <button
                      onClick={startPoseCaptureSequence}
                      disabled={isCountdownActive || isCapturingSequence}
                      className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                        isCountdownActive || isCapturingSequence
                          ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                          : 'bg-emerald-500 text-slate-900 hover:bg-emerald-400'
                      }`}
                    >
                      Start Timed Capture
                    </button>
                    {poseLibrary[selectedNote] && (
                      <button
                        onClick={() => clearPose(selectedNote)}
                        disabled={isCapturingSequence || isCountdownActive}
                        className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                          isCapturingSequence || isCountdownActive
                            ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                            : 'bg-red-500 text-white hover:bg-red-400'
                        }`}
                      >
                        Clear {selectedNote.toUpperCase()}
                      </button>
                    )}
                  </div>
                  <p className="text-sm text-slate-400">
                    Step back until your whole body fits in frame. The app will count down from five, then capture five samples over the next few seconds so you can pick the clearest one.
                  </p>
                  {captureCandidates && (
                    <div className="space-y-3">
                      <p className="text-sm font-medium text-slate-200">Choose the snapshot that best represents your pose.</p>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                        {captureCandidates.map((candidate, index) => (
                          <button
                            key={candidate.id}
                            type="button"
                            onClick={() => setSelectedCandidateIndex(index)}
                            className={`relative rounded-lg overflow-hidden border-2 focus:outline-none focus:ring-2 focus:ring-emerald-400 ${
                              selectedCandidateIndex === index ? 'border-emerald-400' : 'border-transparent'
                            }`}
                          >
                            <img src={candidate.thumbnail} alt={`Pose sample ${index + 1}`} className="w-full h-32 object-cover" />
                            <span className="absolute bottom-1 right-2 bg-slate-900/70 px-2 py-0.5 text-xs rounded-full">
                              Sample {index + 1}
                            </span>
                          </button>
                        ))}
                      </div>
                      <div className="flex flex-wrap gap-3">
                        <button
                          type="button"
                          onClick={saveSelectedPose}
                          className="px-4 py-2 rounded-lg bg-emerald-500 text-slate-900 font-semibold hover:bg-emerald-400 transition-colors"
                        >
                          Save Pose for {selectedNote.toUpperCase()}
                        </button>
                        <button
                          type="button"
                          onClick={startPoseCaptureSequence}
                          disabled={isCountdownActive || isCapturingSequence}
                          className="px-4 py-2 rounded-lg bg-slate-700 text-slate-200 font-semibold hover:bg-slate-600 transition-colors"
                        >
                          Retake Samples
                        </button>
                      </div>
                      <p className="text-xs text-slate-400">
                        Picking the sharpest snapshot leads to better recognition during performance.
                      </p>
                    </div>
                  )}
                </div>
              )}

              {mode === 'perform' && (
                <p className="text-sm text-slate-400">
                  Hold one of your trained poses steady to trigger its musical note. Try transitioning smoothly between poses to play melodies.
                </p>
              )}

              <div className="pt-3 border-t border-slate-700">
                <p className="text-sm text-slate-300">{statusMessage}</p>
                {errorMessage && <p className="text-sm text-red-400 mt-2">{errorMessage}</p>}
                {isLoadingDetector && <p className="text-sm text-slate-400 mt-2">Loading MoveNet model...</p>}
              </div>
            </div>
          </div>

          <div className="w-full lg:w-72 bg-slate-800 rounded-xl p-5 space-y-4 h-fit">
            <h2 className="text-xl font-semibold">Pose Library</h2>
            <p className="text-sm text-slate-400">
              Capture unique poses for each solfège note. Each saved pose will be used during performance mode for matching.
            </p>
            <ul className="space-y-2">
              {NOTES.map((note) => (
                <li
                  key={note}
                  className={`flex items-center justify-between px-3 py-2 rounded-lg border ${
                    poseLibrary[note] ? 'border-emerald-500 bg-emerald-500/10' : 'border-slate-700'
                  }`}
                >
                  <span className="font-medium">{note.toUpperCase()}</span>
                  <span className="text-xs uppercase tracking-wide text-slate-400">
                    {poseLibrary[note] ? 'Captured' : 'Not set'}
                  </span>
                </li>
              ))}
            </ul>
            <div className="text-sm text-slate-400 border-t border-slate-700 pt-3">
              Saved poses: {poseLibraryCount} / {NOTES.length}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function captureFrameThumbnail(video: HTMLVideoElement): string | null {
  const canvas = document.createElement('canvas');
  const width = video.videoWidth;
  const height = video.videoHeight;

  if (width === 0 || height === 0) {
    return null;
  }

  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) {
    return null;
  }

  context.drawImage(video, 0, 0, width, height);
  return canvas.toDataURL('image/jpeg', 0.85);
}

function normalizeKeypoints(keypoints: poseDetection.Keypoint[]): NormalizedKeypoint[] | null {
  const confident = keypoints.filter((keypoint) => (keypoint.score ?? 0) > 0.3);
  if (confident.length === 0) {
    return null;
  }

  const centerX = confident.reduce((sum, point) => sum + point.x, 0) / confident.length;
  const centerY = confident.reduce((sum, point) => sum + point.y, 0) / confident.length;

  let maxDistance = 0;
  confident.forEach((point) => {
    const dx = point.x - centerX;
    const dy = point.y - centerY;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance > maxDistance) {
      maxDistance = distance;
    }
  });

  const scale = maxDistance || 1;

  return keypoints.map((keypoint) => ({
    x: (keypoint.x - centerX) / scale,
    y: (keypoint.y - centerY) / scale,
    score: keypoint.score,
  }));
}

function classifyPose(pose: poseDetection.Pose, library: PoseLibrary): string | null {
  const normalizedCandidate = normalizeKeypoints(pose.keypoints);
  if (!normalizedCandidate) {
    return null;
  }

  let bestNote: string | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const [note, savedKeypoints] of Object.entries(library)) {
    const distance = calculatePoseDistance(normalizedCandidate, savedKeypoints);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestNote = note;
    }
  }

  if (bestNote && bestDistance <= MATCH_THRESHOLD) {
    return bestNote;
  }

  return null;
}

function calculatePoseDistance(
  candidate: NormalizedKeypoint[],
  reference: NormalizedKeypoint[]
): number {
  if (candidate.length !== reference.length) {
    return Number.POSITIVE_INFINITY;
  }

  let totalDistance = 0;
  let usedCount = 0;

  for (let i = 0; i < candidate.length; i += 1) {
    const candidateScore = candidate[i].score ?? 0;
    const referenceScore = reference[i].score ?? 0;

    if (candidateScore < 0.2 && referenceScore < 0.2) {
      continue;
    }

    const dx = candidate[i].x - reference[i].x;
    const dy = candidate[i].y - reference[i].y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    totalDistance += distance;
    usedCount += 1;
  }

  if (usedCount === 0) {
    return Number.POSITIVE_INFINITY;
  }

  return totalDistance / usedCount;
}
