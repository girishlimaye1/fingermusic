import { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection';

type SavedPose = {
  keypoints: Array<{ x: number; y: number; score?: number }>;
};

type PoseLibrary = {
  [key: string]: SavedPose;
};

const NOTES = ['do', 're', 'mi', 'fa', 'so', 'la', 'ti'];

export default function Home() {
  const [screen, setScreen] = useState<'setup' | 'play'>('setup');
  const [poseLibrary, setPoseLibrary] = useState<PoseLibrary>({});
  const [currentNote, setCurrentNote] = useState<string>('');
  const [selectedNote, setSelectedNote] = useState<string>('do');
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [detectedNote, setDetectedNote] = useState<string>('');

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const detectorRef = useRef<poseDetection.PoseDetector | null>(null);
  const animationRef = useRef<number>();
  const lastPlayedRef = useRef<{ note: string; time: number }>({ note: '', time: 0 });

  // Initialize pose detector
  useEffect(() => {
    const initDetector = async () => {
      try {
        await tf.ready();
        await tf.setBackend('webgl');

        const detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
        );

        detectorRef.current = detector;
        setIsModelLoading(false);
      } catch (error) {
        console.error('Failed to load model:', error);
        alert('Failed to load pose detection model. Please refresh the page.');
      }
    };

    initDetector();

    return () => {
      detectorRef.current?.dispose();
    };
  }, []);

  // Start camera
  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 640 },
            height: { ideal: 480 }
          }
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error('Camera error:', error);
        alert('Please allow camera access to use this app.');
      }
    };

    startCamera();

    return () => {
      const stream = videoRef.current?.srcObject as MediaStream;
      stream?.getTracks().forEach(track => track.stop());
    };
  }, []);

  // Pose detection loop
  useEffect(() => {
    const detectPose = async () => {
      if (!detectorRef.current || !videoRef.current || !canvasRef.current) {
        animationRef.current = requestAnimationFrame(detectPose);
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      if (video.readyState === 4) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const poses = await detectorRef.current.estimatePoses(video);

        ctx?.clearRect(0, 0, canvas.width, canvas.height);

        if (poses.length > 0) {
          const pose = poses[0];
          drawPose(pose, ctx!);

          if (screen === 'play') {
            const matched = matchPose(pose);
            if (matched) {
              setDetectedNote(matched);
              playNote(matched);
            } else {
              setDetectedNote('');
            }
          }
        }
      }

      animationRef.current = requestAnimationFrame(detectPose);
    };

    detectPose();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [screen, poseLibrary]);

  const drawPose = (pose: poseDetection.Pose, ctx: CanvasRenderingContext2D) => {
    // Draw keypoints
    pose.keypoints.forEach((kp) => {
      if ((kp.score || 0) > 0.3) {
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 8, 0, 2 * Math.PI);
        ctx.fillStyle = '#10b981';
        ctx.fill();
      }
    });

    // Draw skeleton
    const adjacentPairs = poseDetection.util.getAdjacentPairs(
      poseDetection.SupportedModels.MoveNet
    );

    adjacentPairs.forEach(([i, j]) => {
      const kp1 = pose.keypoints[i];
      const kp2 = pose.keypoints[j];

      if ((kp1.score || 0) > 0.3 && (kp2.score || 0) > 0.3) {
        ctx.beginPath();
        ctx.moveTo(kp1.x, kp1.y);
        ctx.lineTo(kp2.x, kp2.y);
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 3;
        ctx.stroke();
      }
    });
  };

  const normalizePose = (keypoints: poseDetection.Keypoint[]) => {
    const validPoints = keypoints.filter(kp => (kp.score || 0) > 0.3);
    if (validPoints.length === 0) return null;

    const centerX = validPoints.reduce((sum, kp) => sum + kp.x, 0) / validPoints.length;
    const centerY = validPoints.reduce((sum, kp) => sum + kp.y, 0) / validPoints.length;

    const maxDist = Math.max(
      ...validPoints.map(kp =>
        Math.sqrt((kp.x - centerX) ** 2 + (kp.y - centerY) ** 2)
      )
    ) || 1;

    return keypoints.map(kp => ({
      x: (kp.x - centerX) / maxDist,
      y: (kp.y - centerY) / maxDist,
      score: kp.score
    }));
  };

  const capturePose = async () => {
    if (!detectorRef.current || !videoRef.current) return;

    const poses = await detectorRef.current.estimatePoses(videoRef.current);
    if (poses.length === 0) {
      alert('No pose detected! Please stand in front of the camera.');
      return;
    }

    const normalized = normalizePose(poses[0].keypoints);
    if (!normalized) {
      alert('Pose quality too low. Try again!');
      return;
    }

    setPoseLibrary(prev => ({
      ...prev,
      [selectedNote]: { keypoints: normalized }
    }));

    setCurrentNote(`Captured pose for ${selectedNote.toUpperCase()}!`);
    setTimeout(() => setCurrentNote(''), 2000);
  };

  const matchPose = (pose: poseDetection.Pose): string | null => {
    const normalized = normalizePose(pose.keypoints);
    if (!normalized) return null;

    let bestMatch: string | null = null;
    let bestScore = Infinity;

    Object.entries(poseLibrary).forEach(([note, saved]) => {
      let distance = 0;
      let count = 0;

      normalized.forEach((kp, i) => {
        const savedKp = saved.keypoints[i];
        if ((kp.score || 0) > 0.3 && (savedKp.score || 0) > 0.3) {
          distance += Math.sqrt(
            (kp.x - savedKp.x) ** 2 + (kp.y - savedKp.y) ** 2
          );
          count++;
        }
      });

      if (count > 0) {
        const avgDistance = distance / count;
        if (avgDistance < bestScore) {
          bestScore = avgDistance;
          bestMatch = note;
        }
      }
    });

    return bestScore < 0.35 ? bestMatch : null;
  };

  const playNote = (note: string) => {
    const now = Date.now();
    if (lastPlayedRef.current.note === note && now - lastPlayedRef.current.time < 500) {
      return;
    }

    const audio = new Audio(`/notes/${note}.mp3`);
    audio.play().catch(() => {
      // Fallback - use Web Audio API to generate tone
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const frequencies: { [key: string]: number } = {
        do: 261.63, re: 293.66, mi: 329.63, fa: 349.23,
        so: 392.00, la: 440.00, ti: 493.88
      };

      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      oscillator.frequency.value = frequencies[note];
      oscillator.type = 'sine';

      gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);

      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.5);
    });

    lastPlayedRef.current = { note, time: now };
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-500 via-pink-500 to-orange-400">
      <div className="container mx-auto px-4 py-6 max-w-2xl">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-2">
            Pose Music
          </h1>
          <p className="text-white/90 text-lg">
            {screen === 'setup' ? 'Create your poses!' : 'Play music with your body!'}
          </p>
        </div>

        {/* Video Feed */}
        <div className="relative bg-black rounded-3xl overflow-hidden shadow-2xl mb-6">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-auto transform -scale-x-100"
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full transform -scale-x-100"
          />

          {isModelLoading && (
            <div className="absolute inset-0 bg-black/80 flex items-center justify-center">
              <div className="text-white text-xl">Loading...</div>
            </div>
          )}

          {detectedNote && screen === 'play' && (
            <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-white/90 px-8 py-4 rounded-full">
              <div className="text-4xl font-bold text-purple-600">
                {detectedNote.toUpperCase()}
              </div>
            </div>
          )}
        </div>

        {/* Setup Screen */}
        {screen === 'setup' && (
          <div className="bg-white/95 backdrop-blur rounded-3xl p-6 shadow-2xl">
            <h2 className="text-2xl font-bold text-purple-600 mb-4">
              Pick a Note
            </h2>

            <div className="grid grid-cols-4 gap-2 mb-6">
              {NOTES.map(note => (
                <button
                  key={note}
                  onClick={() => setSelectedNote(note)}
                  className={`py-3 px-2 rounded-xl font-bold text-lg transition-all ${
                    selectedNote === note
                      ? 'bg-purple-600 text-white scale-105'
                      : poseLibrary[note]
                      ? 'bg-green-500 text-white'
                      : 'bg-gray-200 text-gray-700'
                  }`}
                >
                  {note.toUpperCase()}
                </button>
              ))}
            </div>

            <button
              onClick={capturePose}
              disabled={isModelLoading}
              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-4 rounded-2xl font-bold text-xl mb-4 active:scale-95 transition-transform disabled:opacity-50"
            >
              Capture {selectedNote.toUpperCase()}
            </button>

            {currentNote && (
              <div className="text-center text-green-600 font-semibold mb-4">
                {currentNote}
              </div>
            )}

            <button
              onClick={() => setScreen('play')}
              disabled={Object.keys(poseLibrary).length === 0}
              className="w-full bg-gradient-to-r from-orange-500 to-red-500 text-white py-4 rounded-2xl font-bold text-xl active:scale-95 transition-transform disabled:opacity-50 disabled:grayscale"
            >
              Start Playing! ({Object.keys(poseLibrary).length}/7)
            </button>

            <p className="text-sm text-gray-600 mt-4 text-center">
              Strike a unique pose for each note, then tap capture!
            </p>
          </div>
        )}

        {/* Play Screen */}
        {screen === 'play' && (
          <div className="bg-white/95 backdrop-blur rounded-3xl p-6 shadow-2xl">
            <h2 className="text-2xl font-bold text-purple-600 mb-4 text-center">
              Move to Play Music!
            </h2>

            <div className="grid grid-cols-4 gap-2 mb-6">
              {NOTES.map(note => (
                <div
                  key={note}
                  className={`py-3 px-2 rounded-xl font-bold text-center transition-all ${
                    poseLibrary[note]
                      ? detectedNote === note
                        ? 'bg-yellow-400 text-purple-900 scale-110'
                        : 'bg-green-500 text-white'
                      : 'bg-gray-200 text-gray-400'
                  }`}
                >
                  {note.toUpperCase()}
                </div>
              ))}
            </div>

            <button
              onClick={() => setScreen('setup')}
              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-4 rounded-2xl font-bold text-xl active:scale-95 transition-transform"
            >
              Back to Setup
            </button>

            <p className="text-sm text-gray-600 mt-4 text-center">
              Hold your saved poses to play the notes!
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
