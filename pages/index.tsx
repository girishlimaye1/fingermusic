import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as handpose from '@tensorflow-models/handpose';

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [note, setNote] = useState<string>('');
  const [model, setModel] = useState<handpose.HandPose | null>(null);

  useEffect(() => {
    const loadModel = async () => {
      const net = await handpose.load();
      setModel(net);
      console.log('Handpose model loaded.');
    };
    loadModel();
  }, []);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing the camera:", err);
      }
    };

    startCamera();

    return () => {
      const stream = videoRef.current?.srcObject as MediaStream;
      const tracks = stream?.getTracks();
      tracks?.forEach(track => track.stop());
    };
  }, []);

  useEffect(() => {
    if (!model) return;

    const detectHands = async () => {
      if (videoRef.current && canvasRef.current) {
        const video = videoRef.current;
        const { videoWidth, videoHeight } = video;
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;

        const hands = await model.estimateHands(video);
        if (hands.length > 0) {
          const fingers = countFingers(hands[0].landmarks);
          const newNote = getNote(fingers);
          if (newNote !== note) {
            setNote(newNote);
            playNote(newNote);
          }
        }
      }
    };

    const interval = setInterval(detectHands, 100);
    return () => clearInterval(interval);
  }, [model, note]);

  const countFingers = (landmarks: number[][]) => {
    const fingerTips = [4, 8, 12, 16, 20];
    let count = 0;
    for (let i = 0; i < fingerTips.length; i++) {
      const tip = fingerTips[i];
      const base = tip - 2;
      if (landmarks[tip][1] < landmarks[base][1]) {
        count++;
      }
    }
    return count;
  };

  const getNote = (fingerCount: number) => {
    const notes = ['do', 're', 'mi', 'fa', 'so', 'la', 'ti'];
    return notes[fingerCount] || 'ti';
  };

  const playNote = (note: string) => {
    const audio = new Audio(`/notes/${note}.mp3`);
    audio.play();
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen py-2">
      <h1 className="text-4xl font-bold mb-4">Finger Music App</h1>
      <div className="relative">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="rounded-lg shadow-lg"
          style={{ width: '640px', height: '480px' }}
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0"
          style={{ width: '640px', height: '480px' }}
        />
      </div>
      <div className="mt-4 text-2xl font-semibold">
        Current Note: {note || 'None'}
      </div>
    </div>
  );
}
