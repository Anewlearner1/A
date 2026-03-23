import * as React from 'react';
import { useEffect, useRef, useState, useCallback, Component } from 'react';
import { Pose, POSE_CONNECTIONS, Results } from '@mediapipe/pose';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { 
  Activity, 
  Upload, 
  Play, 
  Pause, 
  RotateCcw, 
  CheckCircle2, 
  Info,
  ChevronRight,
  FileText,
  Zap,
  Move,
  Timer,
  Ruler,
  History,
  LogOut,
  LogIn,
  Download,
  X,
  Calendar,
  ExternalLink
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Legend
} from 'recharts';
import { useDropzone } from 'react-dropzone';
import { GoogleGenAI } from "@google/genai";
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { auth, signInWithGoogle, logout, saveUserProfile, saveAnalysisRecord, getUserHistory, uploadVideo } from './firebase';
import { onAuthStateChanged, User } from 'firebase/auth';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

// --- Error Handling ---
enum OperationType {
  CREATE = 'create',
  UPDATE = 'update',
  DELETE = 'delete',
  LIST = 'list',
  GET = 'get',
  WRITE = 'write',
}

interface FirestoreErrorInfo {
  error: string;
  operationType: OperationType;
  path: string | null;
  authInfo: {
    userId: string | undefined;
    email: string | null | undefined;
    emailVerified: boolean | undefined;
    isAnonymous: boolean | undefined;
    tenantId: string | null | undefined;
    providerInfo: {
      providerId: string;
      displayName: string | null;
      email: string | null;
      photoUrl: string | null;
    }[];
  }
}

function handleFirestoreError(error: unknown, operationType: OperationType, path: string | null) {
  const errInfo: FirestoreErrorInfo = {
    error: error instanceof Error ? error.message : String(error),
    authInfo: {
      userId: auth.currentUser?.uid,
      email: auth.currentUser?.email,
      emailVerified: auth.currentUser?.emailVerified,
      isAnonymous: auth.currentUser?.isAnonymous,
      tenantId: auth.currentUser?.tenantId,
      providerInfo: auth.currentUser?.providerData.map(provider => ({
        providerId: provider.providerId,
        displayName: provider.displayName,
        email: provider.email,
        photoUrl: provider.photoURL
      })) || []
    },
    operationType,
    path
  }
  console.error('Firestore Error: ', JSON.stringify(errInfo));
  throw new Error(JSON.stringify(errInfo));
}

class ErrorBoundary extends Component<{ children: React.ReactNode }, { hasError: boolean, errorInfo: string | null }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, errorInfo: null };
  }

  static getDerivedStateFromError(error: any) {
    return { hasError: true, errorInfo: error.message };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#0A0A0A] flex items-center justify-center p-6">
          <div className="max-w-md w-full bg-white/5 border border-red-500/20 rounded-3xl p-8 text-center">
            <Zap className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <h2 className="text-xl font-bold mb-2">應用程式發生錯誤</h2>
            <p className="text-sm text-white/40 mb-6">請嘗試重新整理頁面。如果問題持續發生，請聯繫支援團隊。</p>
            <div className="bg-black/40 p-4 rounded-xl text-left mb-6 overflow-hidden">
              <p className="text-[10px] font-mono text-red-400 break-all">{this.state.errorInfo}</p>
            </div>
            <button 
              onClick={() => window.location.reload()}
              className="w-full py-3 bg-emerald-500 text-black rounded-xl font-bold hover:bg-emerald-400 transition-all"
            >
              重新整理
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// --- Utility ---
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Types ---
interface AngleData {
  time: number;
  leftKnee: number;
  rightKnee: number;
  leftHip: number;
  rightHip: number;
  leftAnkle: number;
  rightAnkle: number;
  trunk: number;
  head: number;
  comX: number;
  comY: number;
  leftPhase: 'Stance' | 'Swing';
  rightPhase: 'Stance' | 'Swing';
  shoulderTilt: number;
  hipTilt: number;
  postureScore: number;
  copX?: number;
  copY?: number;
}

interface GaitMetrics {
  strideLength: number;
  leftAnkleAngle: number;
  rightAnkleAngle: number;
  trunkAngle: number;
  headAngle: number;
}

interface KeyPhase {
  type: 'Initial Contact' | 'Mid-stance' | 'Push-off';
  time: number;
  metrics: any;
  thumbnail: string;
}

// --- Helper: Calculate Angle ---
const calculateAngle = (a: any, b: any, c: any) => {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180.0) / Math.PI);
  if (angle > 180.0) angle = 360 - angle;
  return angle;
};

const calculateVerticalAngle = (p1: any, p2: any) => {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const radians = Math.atan2(dx, dy);
  let angle = Math.abs((radians * 180.0) / Math.PI);
  // We want the deviation from vertical (0 degrees is perfectly upright)
  return angle;
};

const calculateHorizontalAngle = (p1: any, p2: any) => {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const radians = Math.atan2(dy, dx);
  return Math.abs((radians * 180.0) / Math.PI);
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'gait' | 'posture'>('gait');
  const [user, setUser] = useState<User | null>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [compareRecord, setCompareRecord] = useState<any | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [alerts, setAlerts] = useState<string[]>([]);
  const isProcessingRef = useRef(false);
  const resultsRef = useRef<Results | null>(null);
  const [angleHistory, setAngleHistory] = useState<AngleData[]>([]);
  const [gaitMetrics, setGaitMetrics] = useState<GaitMetrics>({
    strideLength: 0,
    leftAnkleAngle: 0,
    rightAnkleAngle: 0,
    trunkAngle: 0,
    headAngle: 0
  });
  const [aiAnalysis, setAiAnalysis] = useState<string | null>(null);
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string | null>(null);
  const [groundLevel, setGroundLevel] = useState<number | null>(null);
  const groundLevelRef = useRef<number | null>(null);
  const calibrationFramesRef = useRef(0);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [keyPhases, setKeyPhases] = useState<KeyPhase[]>([]);
  const keyPhasesRef = useRef<KeyPhase[]>([]);
  const lastProcessedTimeRef = useRef<number>(-1);
  const [copHistory, setCopHistory] = useState<{x: number, y: number}[]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseRef = useRef<Pose | null>(null);
  const requestRef = useRef<number | null>(null);
  
  // Tracking for gait parameters
  const lastStepTimeRef = useRef<number>(0);
  const stepTimesRef = useRef<number[]>([]);
  const footStatesRef = useRef({ left: 'Swing', right: 'Swing' });
  const doubleSupportFramesRef = useRef(0);
  const totalFramesRef = useRef(0);
  const maxAnkleYRef = useRef(0);

  // --- Initialize MediaPipe Pose ---
  useEffect(() => {
    let isMounted = true;
    const pose = new Pose({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`;
      },
    });

    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    pose.onResults((results) => {
      if (!isMounted) return;
      resultsRef.current = results;
      isProcessingRef.current = false;

      if (results.poseLandmarks && videoRef.current) {
        const landmarks = results.poseLandmarks;
        const currentTime = videoRef.current.currentTime;
        totalFramesRef.current++;

        // Basic Angles
        const leftKnee = calculateAngle(landmarks[23], landmarks[25], landmarks[27]);
        const rightKnee = calculateAngle(landmarks[24], landmarks[26], landmarks[28]);
        const leftHip = calculateAngle(landmarks[11], landmarks[23], landmarks[25]);
        const rightHip = calculateAngle(landmarks[12], landmarks[24], landmarks[26]);
        const leftAnkle = calculateAngle(landmarks[25], landmarks[27], landmarks[31]);
        const rightAnkle = calculateAngle(landmarks[26], landmarks[28], landmarks[32]);

        // Trunk & Head
        const midShoulder = {
          x: (landmarks[11].x + landmarks[12].x) / 2,
          y: (landmarks[11].y + landmarks[12].y) / 2
        };
        const midHip = {
          x: (landmarks[23].x + landmarks[24].x) / 2,
          y: (landmarks[23].y + landmarks[24].y) / 2
        };
        const midEar = {
          x: (landmarks[7].x + landmarks[8].x) / 2,
          y: (landmarks[7].y + landmarks[8].y) / 2
        };
        const trunk = calculateVerticalAngle(midShoulder, midHip);
        const head = calculateVerticalAngle(midEar, midShoulder);

        // Posture Specifics
        const shoulderTilt = calculateHorizontalAngle(landmarks[11], landmarks[12]);
        const hipTilt = calculateHorizontalAngle(landmarks[23], landmarks[24]);
        
        // Anomaly Detection
        const currentAlerts: string[] = [];
        if (Math.abs(hipTilt) > 6) currentAlerts.push("偵測到骨盆傾斜 (Pelvic Tilt)");
        
        // Toe-in / Toe-out Heuristic (Front/Back view)
        const lFootAngle = calculateHorizontalAngle(landmarks[27], landmarks[31]);
        const rFootAngle = calculateHorizontalAngle(landmarks[28], landmarks[32]);
        // If toes point significantly towards each other or away
        if (lFootAngle > 100 || rFootAngle < 80) currentAlerts.push("疑似內八 (Internal Rotation)");
        if (lFootAngle < 80 || rFootAngle > 100) currentAlerts.push("疑似外八 (External Rotation)");
        
        setAlerts(Array.from(new Set(currentAlerts)));

        // Posture Score (0-100)
        // Penalize deviations from 0 (neutral)
        const trunkPenalty = Math.min(trunk * 2, 30);
        const headPenalty = Math.min(head * 2, 30);
        const shoulderPenalty = Math.min(shoulderTilt * 4, 20);
        const hipPenalty = Math.min(hipTilt * 4, 20);
        const postureScore = Math.max(0, 100 - (trunkPenalty + headPenalty + shoulderPenalty + hipPenalty));

        // Center of Mass (CoM) Approximation
        const comX = (midShoulder.x + midHip.x) / 2;
        const comY = (midShoulder.y + midHip.y) / 2;

        // Automatic Calibration: Find ground level from lowest ankle points
        if (calibrationFramesRef.current < 60) {
          const currentLowest = Math.max(landmarks[27].y, landmarks[28].y, landmarks[31].y, landmarks[32].y);
          const newGround = groundLevelRef.current === null ? currentLowest : Math.max(groundLevelRef.current, currentLowest);
          groundLevelRef.current = newGround;
          setGroundLevel(newGround);
          calibrationFramesRef.current++;
          setCalibrationProgress(calibrationFramesRef.current);
        }

        // Gait Cycle Detection (Stance vs Swing)
        const lAnkleY = landmarks[27].y;
        const rAnkleY = landmarks[28].y;
        const groundRef = groundLevelRef.current || 0.9; // Use ref to avoid closure issue
        
        const threshold = 0.03; 
        const leftPhase = (groundRef - lAnkleY) < threshold ? 'Stance' : 'Swing';
        const rightPhase = (groundRef - rAnkleY) < threshold ? 'Stance' : 'Swing';

        // COP Estimation (Center of Pressure)
        // Weighted average of foot landmarks when in stance
        let copX = 0;
        let copY = 0;
        let stanceCount = 0;
        if (leftPhase === 'Stance') {
          copX += (landmarks[27].x + landmarks[31].x) / 2;
          copY += (landmarks[27].y + landmarks[31].y) / 2;
          stanceCount++;
        }
        if (rightPhase === 'Stance') {
          copX += (landmarks[28].x + landmarks[32].x) / 2;
          copY += (landmarks[28].y + landmarks[32].y) / 2;
          stanceCount++;
        }
        if (stanceCount > 0) {
          copX /= stanceCount;
          copY /= stanceCount;
          setCopHistory(prev => [...prev, {x: copX, y: copY}].slice(-50));
        }

        // Key Phase Detection
        const capturePhase = (type: KeyPhase['type']) => {
          // Use ref to avoid closure issue
          if (keyPhasesRef.current.some(p => p.type === type && Math.abs(p.time - currentTime) < 1)) return;
          
          const canvas = canvasRef.current;
          if (canvas) {
            const thumbnail = canvas.toDataURL('image/jpeg', 0.5);
            const newPhase = {
              type,
              time: currentTime,
              metrics: { leftKnee, rightKnee, trunk, head },
              thumbnail
            };
            keyPhasesRef.current = [...keyPhasesRef.current, newPhase].slice(-6);
            setKeyPhases(keyPhasesRef.current);
          }
        };

        // 1. Initial Contact (Heel Strike)
        if (leftPhase === 'Stance' && footStatesRef.current.left === 'Swing') {
          capturePhase('Initial Contact');
          stepTimesRef.current.push(currentTime);
          footStatesRef.current.left = 'Stance';
        } else if (leftPhase === 'Swing') {
          footStatesRef.current.left = 'Swing';
        }

        // 2. Mid-stance (CoM over stance foot)
        if (leftPhase === 'Stance' && Math.abs(comX - landmarks[27].x) < 0.02) {
          capturePhase('Mid-stance');
        }

        // 3. Push-off (Toe off)
        if (leftPhase === 'Swing' && footStatesRef.current.left === 'Stance') {
          capturePhase('Push-off');
        }

        if (rightPhase === 'Stance' && footStatesRef.current.right === 'Swing') {
          stepTimesRef.current.push(currentTime);
          footStatesRef.current.right = 'Stance';
        } else if (rightPhase === 'Swing') {
          footStatesRef.current.right = 'Swing';
        }

        // Double Support Tracking
        if (leftPhase === 'Stance' && rightPhase === 'Stance') {
          doubleSupportFramesRef.current++;
        }

        // Update Metrics
        setGaitMetrics({
          strideLength: 0, // Requires calibration, keeping as 0 for now or relative
          leftAnkleAngle: Math.round(leftAnkle),
          rightAnkleAngle: Math.round(rightAnkle),
          trunkAngle: Math.round(trunk),
          headAngle: Math.round(head)
        });

        setAngleHistory(prev => {
          const newData = [...prev, {
            time: Number(currentTime.toFixed(2)),
            leftKnee,
            rightKnee,
            leftHip,
            rightHip,
            leftAnkle,
            rightAnkle,
            trunk,
            head,
            comX,
            comY,
            leftPhase: leftPhase as any,
            rightPhase: rightPhase as any,
            shoulderTilt,
            hipTilt,
            postureScore,
            copX,
            copY
          }].slice(-100);
          return newData;
        });
      }
    });

    poseRef.current = pose;

    return () => {
      isMounted = false;
      if (poseRef.current) {
        poseRef.current.close();
      }
    };
  }, []);

  // --- Auth & History ---
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      if (currentUser) {
        saveUserProfile(currentUser);
      } else {
        setHistory([]);
      }
    });
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    if (user) {
      const unsubscribe = getUserHistory(user.uid, activeTab, (records) => {
        setHistory(records);
      });
      return () => unsubscribe();
    }
  }, [user, activeTab]);

  // --- Drawing Loop ---
  const draw = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !poseRef.current) return;

    if (videoRef.current.ended) return;
    if (videoRef.current.videoWidth === 0) {
      requestRef.current = requestAnimationFrame(draw);
      return;
    }

    const canvasCtx = canvasRef.current.getContext('2d');
    if (!canvasCtx) return;

    // Ensure canvas dimensions match video natural dimensions
    if (canvasRef.current.width !== videoRef.current.videoWidth || canvasRef.current.height !== videoRef.current.videoHeight) {
      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    
    // Draw video frame
    canvasCtx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);

    // Process frame if playing OR if the frame has changed (for manual stepping)
    const currentTime = videoRef.current.currentTime;
    const hasFrameChanged = Math.abs(currentTime - lastProcessedTimeRef.current) > 0.001;

    if (!isProcessingRef.current && (isPlaying || hasFrameChanged)) {
      isProcessingRef.current = true;
      lastProcessedTimeRef.current = currentTime;
      try {
        await poseRef.current.send({ image: videoRef.current });
      } catch (err) {
        console.error("Pose processing error:", err);
        isProcessingRef.current = false;
      }
    }

    const currentResults = resultsRef.current;
    if (currentResults?.poseLandmarks) {
      drawConnectors(canvasCtx, currentResults.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
      drawLandmarks(canvasCtx, currentResults.poseLandmarks, { color: '#FF0000', lineWidth: 1, radius: 3 });

      const landmarks = currentResults.poseLandmarks;
      const midShoulder = {
        x: (landmarks[11].x + landmarks[12].x) / 2,
        y: (landmarks[11].y + landmarks[12].y) / 2
      };
      const midHip = {
        x: (landmarks[23].x + landmarks[24].x) / 2,
        y: (landmarks[23].y + landmarks[24].y) / 2
      };
      const midEar = {
        x: (landmarks[7].x + landmarks[8].x) / 2,
        y: (landmarks[7].y + landmarks[8].y) / 2
      };

      const joints = [
        { id: 'L-Knee', p1: 23, p2: 25, p3: 27, type: 'joint' },
        { id: 'R-Knee', p1: 24, p2: 26, p3: 28, type: 'joint' },
        { id: 'L-Hip', p1: 11, p2: 23, p3: 25, type: 'joint' },
        { id: 'R-Hip', p1: 12, p2: 24, p3: 26, type: 'joint' },
        { id: 'Trunk', p1: midShoulder, p2: midHip, type: 'vertical' },
        { id: 'Head', p1: midEar, p2: midShoulder, type: 'vertical' },
      ];

      joints.forEach(joint => {
        let angle = 0;
        let pos = { x: 0, y: 0 };

        if (joint.type === 'joint') {
          angle = calculateAngle(landmarks[joint.p1 as number], landmarks[joint.p2 as number], landmarks[joint.p3 as number]);
          pos = landmarks[joint.p2 as number];
        } else {
          angle = calculateVerticalAngle(joint.p1, joint.p2);
          pos = joint.p2 as any;
        }
        
        canvasCtx.font = 'bold 16px Inter';
        canvasCtx.fillStyle = '#00FF00';
        canvasCtx.fillText(`${Math.round(angle)}°`, pos.x * canvasRef.current!.width, pos.y * canvasRef.current!.height - 10);
      });

      // Draw Posture Lines (Shoulders and Hips)
      canvasCtx.beginPath();
      canvasCtx.setLineDash([5, 5]);
      canvasCtx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      canvasCtx.moveTo(landmarks[11].x * canvasRef.current!.width, landmarks[11].y * canvasRef.current!.height);
      canvasCtx.lineTo(landmarks[12].x * canvasRef.current!.width, landmarks[12].y * canvasRef.current!.height);
      canvasCtx.stroke();
      canvasCtx.moveTo(landmarks[23].x * canvasRef.current!.width, landmarks[23].y * canvasRef.current!.height);
      canvasCtx.lineTo(landmarks[24].x * canvasRef.current!.width, landmarks[24].y * canvasRef.current!.height);
      canvasCtx.stroke();
      canvasCtx.setLineDash([]);

      // Draw Gait Phase Labels
      const lastData = angleHistory[angleHistory.length - 1];
      if (lastData) {
        canvasCtx.font = 'bold 14px Inter';
        canvasCtx.fillStyle = lastData.leftPhase === 'Stance' ? '#10b981' : '#3b82f6';
        canvasCtx.fillText(`L: ${lastData.leftPhase}`, landmarks[27].x * canvasRef.current!.width, landmarks[27].y * canvasRef.current!.height + 20);
        
        canvasCtx.fillStyle = lastData.rightPhase === 'Stance' ? '#10b981' : '#3b82f6';
        canvasCtx.fillText(`R: ${lastData.rightPhase}`, landmarks[28].x * canvasRef.current!.width, landmarks[28].y * canvasRef.current!.height + 20);
      }

      // Draw COP Trace
      if (copHistory.length > 1) {
        canvasCtx.beginPath();
        canvasCtx.strokeStyle = '#FBBF24'; // Amber
        canvasCtx.lineWidth = 3;
        canvasCtx.lineJoin = 'round';
        copHistory.forEach((p, i) => {
          if (i === 0) canvasCtx.moveTo(p.x * canvasRef.current!.width, p.y * canvasRef.current!.height);
          else canvasCtx.lineTo(p.x * canvasRef.current!.width, p.y * canvasRef.current!.height);
        });
        canvasCtx.stroke();
        
        // Current COP point
        const lastCop = copHistory[copHistory.length - 1];
        canvasCtx.beginPath();
        canvasCtx.fillStyle = '#FBBF24';
        canvasCtx.arc(lastCop.x * canvasRef.current!.width, lastCop.y * canvasRef.current!.height, 6, 0, Math.PI * 2);
        canvasCtx.fill();
      }

      // Draw Ground Line
      if (groundLevel) {
        canvasCtx.beginPath();
        canvasCtx.strokeStyle = 'rgba(16, 185, 129, 0.3)';
        canvasCtx.setLineDash([10, 10]);
        canvasCtx.moveTo(0, groundLevel * canvasRef.current!.height);
        canvasCtx.lineTo(canvasRef.current!.width, groundLevel * canvasRef.current!.height);
        canvasCtx.stroke();
        canvasCtx.setLineDash([]);
      }
    }

    canvasCtx.restore();
    requestRef.current = requestAnimationFrame(draw);
  }, []); // Removed results dependency

  useEffect(() => {
    if (videoUrl) {
      requestRef.current = requestAnimationFrame(draw);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [videoUrl, draw]);

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const stepFrame = (frames: number) => {
    if (videoRef.current) {
      videoRef.current.pause();
      setIsPlaying(false);
      videoRef.current.currentTime += frames * (1 / 30); // Assume 30fps
    }
  };

  const changePlaybackRate = (rate: number) => {
    if (videoRef.current) {
      videoRef.current.playbackRate = rate;
      setPlaybackRate(rate);
    }
  };

  // --- File Handling ---
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setUploadedVideoUrl(null);
      setAngleHistory([]);
      setAiAnalysis(null);
      setKeyPhases([]);
      keyPhasesRef.current = [];
      setCopHistory([]);
      setGroundLevel(null);
      groundLevelRef.current = null;
      calibrationFramesRef.current = 0;
      setCalibrationProgress(0);
      lastProcessedTimeRef.current = -1;
      // Reset tracking refs
      stepTimesRef.current = [];
      doubleSupportFramesRef.current = 0;
      totalFramesRef.current = 0;
      maxAnkleYRef.current = 0;
      footStatesRef.current = { left: 'Swing', right: 'Swing' };
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false
  } as any);

  // --- Gemini Analysis ---
  const runAiAnalysis = async () => {
    if (!angleHistory.length) return;
    setIsAiLoading(true);
    try {
      let currentVideoUrl = uploadedVideoUrl;

      // Upload video if user is logged in and not already uploaded
      if (user && videoFile && !currentVideoUrl) {
        setIsUploading(true);
        try {
          currentVideoUrl = await uploadVideo(user.uid, videoFile);
          setUploadedVideoUrl(currentVideoUrl);
        } catch (err) {
          console.error("Video upload failed:", err);
        } finally {
          setIsUploading(false);
        }
      }

      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      const prompt = activeTab === 'gait' 
        ? `Analyze this gait data from a walking video. 
           Data (subset): ${JSON.stringify(angleHistory.filter((_, i) => i % 10 === 0))}
           Please provide the following in Traditional Chinese (繁體中文):
           1. 步行模式的臨床總結。
           2. 識別任何異常（例如：膝蓋彎曲受限、步頻問題、內八/外八、骨盆傾斜）。
           3. 改善建議。
           Keep it professional and concise. Use Markdown formatting.`
        : `Analyze this posture data. 
           Data (subset): ${JSON.stringify(angleHistory.filter((_, i) => i % 10 === 0))}
           Please provide the following in Traditional Chinese (繁體中文):
           1. 整體姿勢與對齊的臨床總結。
           2. 識別任何異常（例如：肩膀傾斜、頭部前傾、軀幹偏離）。
           3. 姿勢評分說明與矯正運動建議。
           Keep it professional and concise. Use Markdown formatting.`;

      const model = ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: prompt,
      });
      const response = await model;
      const resultText = response.text || "Unable to generate analysis.";
      setAiAnalysis(resultText);

      // Auto-save to cloud if user is logged in
      if (user) {
        setIsSaving(true);
        try {
          const metrics = activeTab === 'gait' ? gaitMetrics : { postureScore: angleHistory[angleHistory.length - 1]?.postureScore };
          await saveAnalysisRecord(user.uid, activeTab, metrics, resultText, angleHistory, currentVideoUrl);
        } catch (error) {
          console.error("Failed to save record:", error);
        } finally {
          setIsSaving(false);
        }
      }
    } catch (error) {
      console.error("AI Analysis failed:", error);
      setAiAnalysis("Error generating AI analysis. Please try again.");
    } finally {
      setIsAiLoading(false);
    }
  };

  const exportPDF = async () => {
    const element = document.getElementById('analysis-report');
    if (!element) return;

    const canvas = await html2canvas(element, {
      backgroundColor: '#0A0A0A',
      scale: 2,
    });
    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF('p', 'mm', 'a4');
    const imgProps = pdf.getImageProperties(imgData);
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
    
    pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
    pdf.save(`StepSense_Report_${new Date().toLocaleDateString()}.pdf`);
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-[#0A0A0A] text-white font-sans selection:bg-emerald-500/30">
      {/* Header */}
      <header className="border-b border-white/10 bg-black/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-emerald-500 rounded-xl flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Activity className="text-black w-6 h-6" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-emerald-500">StepSense AI</h1>
              <p className="text-[10px] uppercase tracking-widest text-white/40 font-semibold">AI 步態與姿勢分析系統 v1.0</p>
            </div>
          </div>
          <nav className="flex items-center gap-1 bg-white/5 p-1 rounded-xl border border-white/10">
            <button 
              onClick={() => setActiveTab('gait')}
              className={cn(
                "px-4 py-1.5 rounded-lg text-xs font-bold transition-all",
                activeTab === 'gait' ? "bg-emerald-500 text-black shadow-lg shadow-emerald-500/20" : "text-white/40 hover:text-white"
              )}
            >
              步態分析
            </button>
            <button 
              onClick={() => setActiveTab('posture')}
              className={cn(
                "px-4 py-1.5 rounded-lg text-xs font-bold transition-all",
                activeTab === 'posture' ? "bg-emerald-500 text-black shadow-lg shadow-emerald-500/20" : "text-white/40 hover:text-white"
              )}
            >
              姿勢評分
            </button>
          </nav>
          <div className="flex items-center gap-4">
            {user ? (
              <div className="flex items-center gap-3">
                <button 
                  onClick={() => setIsHistoryOpen(true)}
                  className="p-2 hover:bg-white/10 rounded-full transition-colors relative"
                  title="History"
                >
                  <History className="w-5 h-5 text-white/60" />
                  {history.length > 0 && (
                    <span className="absolute top-1 right-1 w-2 h-2 bg-emerald-500 rounded-full border border-black" />
                  )}
                </button>
                <div className="flex items-center gap-2 pl-3 border-l border-white/10">
                  <img src={user.photoURL || ''} alt="" className="w-8 h-8 rounded-full border border-white/20" />
                  <button onClick={logout} className="p-2 hover:bg-white/10 rounded-full transition-colors text-white/40 hover:text-red-400">
                    <LogOut className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ) : (
              <button 
                onClick={signInWithGoogle}
                className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 rounded-xl border border-white/10 transition-all text-xs font-bold"
              >
                <LogIn className="w-4 h-4" />
                登入
              </button>
            )}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-full border border-white/10">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-xs font-medium text-white/70">系統就緒</span>
            </div>
          </div>
        </div>
      </header>

      <main id="analysis-report" className="max-w-7xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Video & Controls */}
        <div className="lg:col-span-8 space-y-6">
          <AnimatePresence>
            {activeTab === 'gait' && alerts.length > 0 && (
              <motion.div 
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="flex flex-wrap gap-2"
              >
                {alerts.map((alert, i) => (
                  <div key={i} className="flex items-center gap-2 px-3 py-1.5 bg-red-500/10 border border-red-500/20 rounded-full text-[10px] font-bold text-red-400 uppercase tracking-wider">
                    <Zap className="w-3 h-3" />
                    {alert}
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>

          <div className="relative aspect-video bg-black rounded-3xl border border-white/10 overflow-hidden group shadow-2xl">
            {/* Calibration Overlay */}
            {videoUrl && calibrationProgress < 60 && (
              <div className="absolute top-6 left-6 z-20 flex items-center gap-3 bg-black/60 backdrop-blur-md px-4 py-2 rounded-2xl border border-emerald-500/30">
                <div className="w-4 h-4 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
                <span className="text-xs font-bold text-emerald-500 uppercase tracking-widest">地面自動校準中... {Math.round((calibrationProgress/60)*100)}%</span>
              </div>
            )}

            {!videoUrl ? (
              <div 
                {...getRootProps()} 
                className={cn(
                  "absolute inset-0 flex flex-col items-center justify-center cursor-pointer transition-all duration-300",
                  isDragActive ? "bg-emerald-500/10 border-2 border-dashed border-emerald-500" : "hover:bg-white/5"
                )}
              >
                <input {...getInputProps()} />
                <div className="w-16 h-16 bg-white/5 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <Upload className="w-8 h-8 text-emerald-500" />
                </div>
                <p className="text-lg font-medium">拖放步態影片至此</p>
                <p className="text-sm text-white/40 mt-1">支援 MP4, MOV 格式 (最大 50MB)</p>
              </div>
            ) : (
              <>
                <video 
                  ref={videoRef}
                  src={videoUrl}
                  className="hidden"
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onEnded={() => setIsPlaying(false)}
                  onLoadedMetadata={(e) => {
                    const video = e.currentTarget;
                    if (canvasRef.current) {
                      canvasRef.current.width = video.videoWidth;
                      canvasRef.current.height = video.videoHeight;
                    }
                  }}
                  muted
                  playsInline
                />
                <canvas 
                  ref={canvasRef}
                  className="w-full h-full object-contain"
                />
                                {/* Video Overlay Controls */}
                <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/90 via-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="flex flex-col gap-4">
                    {/* Progress Bar (Visual only for now) */}
                    <div className="w-full h-1 bg-white/20 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-emerald-500 transition-all duration-100" 
                        style={{ width: `${(videoRef.current?.currentTime || 0) / (videoRef.current?.duration || 1) * 100}%` }}
                      />
                    </div>

                    <div className="flex items-center justify-between gap-4">
                      <div className="flex items-center gap-6">
                        {/* Playback Speed */}
                        <div className="flex flex-col gap-1">
                          <span className="text-[8px] uppercase tracking-widest text-white/40 font-bold">播放速度</span>
                          <div className="flex items-center gap-1 bg-white/10 p-1 rounded-xl border border-white/10">
                            {[0.25, 0.5, 1].map(rate => (
                              <button
                                key={rate}
                                onClick={() => changePlaybackRate(rate)}
                                className={cn(
                                  "px-2.5 py-1 rounded-lg text-[10px] font-bold transition-all",
                                  playbackRate === rate ? "bg-emerald-500 text-black shadow-lg shadow-emerald-500/20" : "text-white/40 hover:text-white"
                                )}
                              >
                                {rate === 1 ? '正常' : `${rate}x`}
                              </button>
                            ))}
                          </div>
                        </div>

                        {/* Frame Control */}
                        <div className="flex flex-col gap-1">
                          <span className="text-[8px] uppercase tracking-widest text-white/40 font-bold text-center">影格步進</span>
                          <div className="flex items-center gap-2">
                            <button 
                              onClick={() => stepFrame(-1)}
                              className="w-10 h-10 bg-white/10 hover:bg-white/20 rounded-full flex items-center justify-center transition-all text-white/60 hover:text-white"
                              title="上一影格"
                            >
                              <RotateCcw className="w-4 h-4 -scale-x-100" />
                            </button>
                            <button 
                              onClick={togglePlay}
                              className="w-14 h-14 bg-emerald-500 rounded-full flex items-center justify-center hover:scale-105 active:scale-95 transition-all shadow-xl shadow-emerald-500/40"
                            >
                              {isPlaying ? <Pause className="text-black fill-black w-6 h-6" /> : <Play className="text-black fill-black w-6 h-6 ml-1" />}
                            </button>
                            <button 
                              onClick={() => stepFrame(1)}
                              className="w-10 h-10 bg-white/10 hover:bg-white/20 rounded-full flex items-center justify-center transition-all text-white/60 hover:text-white"
                              title="下一影格"
                            >
                              <RotateCcw className="w-4 h-4" />
                            </button>
                          </div>
                        </div>

                        <button 
                          onClick={() => {
                            if (videoRef.current) {
                              videoRef.current.currentTime = 0;
                              setAngleHistory([]);
                            }
                          }}
                          className="w-10 h-10 bg-white/10 rounded-full flex items-center justify-center hover:bg-white/20 transition-colors mt-4"
                          title="重設影片"
                        >
                          <RotateCcw className="w-5 h-5" />
                        </button>
                      </div>
                      
                      <button 
                        onClick={() => {
                          setVideoFile(null);
                          setVideoUrl(null);
                          setAngleHistory([]);
                          setAiAnalysis(null);
                          setAlerts([]);
                        }}
                        className="text-xs font-bold text-white/40 hover:text-red-400 transition-colors uppercase tracking-widest mt-4"
                      >
                        清除影片
                      </button>
                    </div>
                  </div>
                </div>

              </>
            )}
          </div>

          {/* Real-time Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {activeTab === 'gait' ? (
              [
                { label: '左踝角度', value: angleHistory[angleHistory.length - 1]?.leftAnkle, unit: '°' },
                { label: '右踝角度', value: angleHistory[angleHistory.length - 1]?.rightAnkle, unit: '°' },
                { label: '軀幹角度', value: angleHistory[angleHistory.length - 1]?.trunk, unit: '°', icon: Activity },
                { label: '頭部角度', value: angleHistory[angleHistory.length - 1]?.head, unit: '°', icon: Activity },
                { label: '左膝角度', value: angleHistory[angleHistory.length - 1]?.leftKnee, unit: '°' },
                { label: '右膝角度', value: angleHistory[angleHistory.length - 1]?.rightKnee, unit: '°' },
                { label: '左髖角度', value: angleHistory[angleHistory.length - 1]?.leftHip, unit: '°' },
                { label: '右髖角度', value: angleHistory[angleHistory.length - 1]?.rightHip, unit: '°' },
              ].map((stat, i) => (
                <MetricCard key={i} {...stat} />
              ))
            ) : (
              [
                { label: '姿勢總分', value: angleHistory[angleHistory.length - 1]?.postureScore, unit: '/100', icon: CheckCircle2, highlight: true },
                { label: '肩膀傾斜', value: angleHistory[angleHistory.length - 1]?.shoulderTilt, icon: Move },
                { label: '骨盆傾斜', value: angleHistory[angleHistory.length - 1]?.hipTilt, icon: Move },
                { label: '軀幹角度', value: angleHistory[angleHistory.length - 1]?.trunk, icon: Activity },
                { label: '頭部角度', value: angleHistory[angleHistory.length - 1]?.head, icon: Activity },
              ].map((stat, i) => (
                <MetricCard key={i} {...stat} />
              ))
            )}
          </div>
        </div>

        {/* Right Column: Analysis & Charts */}
        <div className="lg:col-span-4 space-y-6">
          {activeTab === 'gait' && (
            <div className="bg-white/5 border border-white/10 rounded-3xl p-6 h-[280px] flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-bold flex items-center gap-2">
                  <Move className="w-4 h-4 text-emerald-500" />
                  重心位移 (CoM Displacement)
                </h3>
                <span className="text-[10px] text-white/40 font-mono">垂直 / 左右</span>
              </div>
              <div className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={angleHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                    <XAxis dataKey="time" hide />
                    <YAxis domain={['auto', 'auto']} hide />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1A1A1A', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px' }}
                      itemStyle={{ fontSize: '10px' }}
                    />
                    <Legend iconType="circle" wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
                    <Line name="垂直位移 (Y)" type="monotone" dataKey="comY" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line name="左右位移 (X)" type="monotone" dataKey="comX" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {activeTab === 'posture' && (
             <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-3xl p-6 flex flex-col gap-4">
                <h3 className="text-sm font-bold flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                  姿勢細節分析
                </h3>
                <div className="space-y-4">
                  <PostureMetric label="脊椎對齊" value={100 - (angleHistory[angleHistory.length - 1]?.trunk * 2 || 0)} />
                  <PostureMetric label="頭部位置" value={100 - (angleHistory[angleHistory.length - 1]?.head * 2 || 0)} />
                  <PostureMetric label="肩膀水平" value={100 - (angleHistory[angleHistory.length - 1]?.shoulderTilt * 4 || 0)} />
                  <PostureMetric label="骨盆水平" value={100 - (angleHistory[angleHistory.length - 1]?.hipTilt * 4 || 0)} />
                </div>
             </div>
          )}

          {/* AI Insights Section */}
          <div className="bg-white/5 border border-white/10 rounded-3xl p-6 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold flex items-center gap-2">
                <Zap className="w-4 h-4 text-emerald-500" />
                關鍵時相自動定格 (Key Phases)
              </h3>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {['Initial Contact', 'Mid-stance', 'Push-off'].map(type => {
                const phase = keyPhases.find(p => p.type === type);
                return (
                  <div key={type} className="flex flex-col gap-2">
                    <div className="aspect-square bg-white/5 rounded-xl overflow-hidden border border-white/10 relative group">
                      {phase ? (
                        <>
                          <img src={phase.thumbnail} alt={type} className="w-full h-full object-cover" />
                          <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                            <button 
                              onClick={() => {
                                if (videoRef.current) videoRef.current.currentTime = phase.time;
                              }}
                              className="p-2 bg-emerald-500 rounded-full text-black"
                            >
                              <Play className="w-3 h-3 fill-black" />
                            </button>
                          </div>
                        </>
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <Timer className="w-4 h-4 text-white/10" />
                        </div>
                      )}
                    </div>
                    <div className="text-center">
                      <p className="text-[9px] font-bold text-white/40 uppercase tracking-tighter">{type}</p>
                      {phase && (
                        <p className="text-[10px] text-emerald-400 font-mono">
                          {Math.round(phase.metrics.leftKnee)}° / {Math.round(phase.metrics.rightKnee)}°
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* AI Insights Section */}
          <div className="bg-white/5 border border-white/10 rounded-3xl p-6 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold flex items-center gap-2">
                <FileText className="w-4 h-4 text-emerald-500" />
                AI 臨床分析報告
              </h3>
              <div className="flex items-center gap-2">
                {aiAnalysis && (
                  <button 
                    onClick={exportPDF}
                    className="p-2 hover:bg-white/10 rounded-full transition-colors text-white/60"
                    title="Export PDF"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                )}
                    {!aiAnalysis && angleHistory.length > 0 && (
                      <button 
                        onClick={runAiAnalysis}
                        disabled={isAiLoading || isUploading}
                        className="text-[10px] bg-emerald-500 text-black px-3 py-1 rounded-full font-bold hover:bg-emerald-400 transition-colors disabled:opacity-50"
                      >
                        {isUploading ? '影片上傳中...' : (isAiLoading ? '分析中...' : '生成報告')}
                      </button>
                    )}
              </div>
            </div>

            <div className="min-h-[200px] text-sm text-white/70 leading-relaxed">
              {isAiLoading ? (
                <div className="flex flex-col items-center justify-center h-full gap-4 py-12">
                  <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
                  <p className="text-xs text-white/40 animate-pulse">Gemini 正在分析步態模式...</p>
                </div>
              ) : aiAnalysis ? (
                <div className="prose prose-invert prose-sm max-w-none">
                  {aiAnalysis.split('\n').map((line, i) => (
                    <p key={i} className="mb-2">{line}</p>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full py-12 text-center opacity-40">
                  <Info className="w-8 h-8 mb-2" />
                  <p className="text-xs">上傳並播放影片以<br />生成 AI 步態分析報告</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer Info */}
      <footer className="max-w-7xl mx-auto px-6 py-8 text-center border-t border-white/5">
        <p className="text-[10px] text-white/20 uppercase tracking-[0.2em]">
          由 Google Gemini & MediaPipe Pose 驅動 • 僅供教育與參考使用
        </p>
      </footer>

      {/* History Modal */}
      <AnimatePresence>
        {isHistoryOpen && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsHistoryOpen(false)}
              className="absolute inset-0 bg-black/80 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="relative w-full max-w-2xl bg-[#1A1A1A] border border-white/10 rounded-3xl overflow-hidden shadow-2xl"
            >
              <div className="p-6 border-b border-white/10 flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-bold">歷史紀錄對比</h2>
                  <p className="text-xs text-white/40 mt-1">查看並對比過去的分析結果</p>
                </div>
                <button 
                  onClick={() => setIsHistoryOpen(false)}
                  className="p-2 hover:bg-white/10 rounded-full transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="p-6 max-h-[60vh] overflow-y-auto space-y-4 custom-scrollbar">
                {history.length === 0 ? (
                  <div className="text-center py-12 opacity-40">
                    <History className="w-12 h-12 mx-auto mb-4" />
                    <p>尚無歷史紀錄</p>
                  </div>
                ) : (
                  history.map((record) => (
                    <div 
                      key={record.id}
                      className={cn(
                        "p-4 rounded-2xl border transition-all cursor-pointer group",
                        compareRecord?.id === record.id ? "bg-emerald-500/10 border-emerald-500/50" : "bg-white/5 border-white/10 hover:border-white/20"
                      )}
                      onClick={() => setCompareRecord(compareRecord?.id === record.id ? null : record)}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 bg-white/5 rounded-xl flex items-center justify-center">
                            <Calendar className="w-5 h-5 text-emerald-500" />
                          </div>
                          <div>
                            <p className="text-sm font-bold">{new Date(record.timestamp).toLocaleDateString()} {new Date(record.timestamp).toLocaleTimeString()}</p>
                            <div className="flex items-center gap-2">
                              <p className="text-[10px] text-white/40 uppercase tracking-wider">{record.type === 'gait' ? '步態分析' : '姿勢評分'}</p>
                              {record.videoUrl && (
                                <a 
                                  href={record.videoUrl} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  onClick={(e) => e.stopPropagation()}
                                  className="flex items-center gap-1 text-[10px] text-emerald-500 hover:underline"
                                >
                                  <ExternalLink className="w-2.5 h-2.5" />
                                  查看影片
                                </a>
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-lg font-mono font-bold text-emerald-400">
                            {record.type === 'gait' ? `${record.metrics.trunkAngle}°` : `${record.metrics.postureScore}/100`}
                          </p>
                        </div>
                      </div>
                      
                      {compareRecord?.id === record.id && (
                        <motion.div 
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          className="pt-4 border-t border-white/10 mt-4 space-y-4"
                        >
                          <div className="grid grid-cols-2 gap-4">
                            <div className="p-3 bg-black/30 rounded-xl">
                              <p className="text-[10px] text-white/40 mb-1 uppercase">歷史數據</p>
                              <p className="text-sm font-bold text-emerald-400">
                                {record.type === 'gait' ? `${record.metrics.trunkAngle}°` : `${record.metrics.postureScore} 分`}
                              </p>
                            </div>
                            <div className="p-3 bg-black/30 rounded-xl">
                              <p className="text-[10px] text-white/40 mb-1 uppercase">當前數據</p>
                              <p className="text-sm font-bold text-blue-400">
                                {record.type === 'gait' ? `${gaitMetrics.trunkAngle}°` : `${angleHistory[angleHistory.length - 1]?.postureScore || 0} 分`}
                              </p>
                            </div>
                          </div>
                          <div className="p-3 bg-black/30 rounded-xl">
                            <p className="text-[10px] text-white/40 mb-2 uppercase">差異對比</p>
                            {(() => {
                              const current = record.type === 'gait' ? gaitMetrics.trunkAngle : (angleHistory[angleHistory.length - 1]?.postureScore || 0);
                              const past = record.type === 'gait' ? record.metrics.trunkAngle : record.metrics.postureScore;
                              const diff = current - past;
                              return (
                                <div className="flex items-center gap-2">
                                  <div className={cn(
                                    "px-2 py-0.5 rounded text-[10px] font-bold",
                                    diff > 0 ? "bg-emerald-500/20 text-emerald-400" : diff < 0 ? "bg-red-500/20 text-red-400" : "bg-white/10 text-white/40"
                                  )}>
                                    {diff > 0 ? '+' : ''}{diff} {record.type === 'gait' ? '°' : '分'}
                                  </div>
                                  <p className="text-xs text-white/60">
                                    {diff > 0 ? '數值增加' : diff < 0 ? '數值減少' : '無明顯變化'}
                                  </p>
                                </div>
                              );
                            })()}
                          </div>
                        </motion.div>
                      )}
                    </div>
                  ))
                )}
              </div>
              
              <div className="p-6 bg-black/20 border-t border-white/10 flex justify-end">
                <button 
                  onClick={() => setIsHistoryOpen(false)}
                  className="px-6 py-2 bg-white/5 hover:bg-white/10 rounded-xl text-sm font-bold transition-all"
                >
                  關閉
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
      </div>
    </ErrorBoundary>
  );
}

// --- Sub-components ---
function MetricCard({ label, value, unit = '°', icon: Icon, highlight = false }: any) {
  const val = Math.round(value || 0);
  return (
    <div className={cn(
      "p-4 rounded-2xl border transition-all",
      highlight ? "bg-emerald-500/10 border-emerald-500/20" : "bg-white/5 border-white/10"
    )}>
      <div className="flex items-center gap-2 mb-2">
        {Icon && <Icon className={cn("w-3 h-3", highlight ? "text-emerald-500" : "text-white/40")} />}
        <span className="text-[10px] font-bold text-white/40 uppercase tracking-widest">{label}</span>
      </div>
      <div className="flex items-baseline gap-1">
        <span className={cn("text-2xl font-mono font-bold", highlight ? "text-emerald-500" : "text-white")}>{val}</span>
        <span className="text-[10px] text-white/20 font-bold">{unit}</span>
      </div>
    </div>
  );
}

function PostureMetric({ label, value }: { label: string, value: number }) {
  const score = Math.round(Math.max(0, Math.min(100, value)));
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-widest">
        <span className="text-white/40">{label}</span>
        <span className={cn(score > 80 ? "text-emerald-500" : score > 60 ? "text-yellow-500" : "text-red-500")}>
          {score}%
        </span>
      </div>
      <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${score}%` }}
          className={cn(
            "h-full transition-all duration-500",
            score > 80 ? "bg-emerald-500" : score > 60 ? "bg-yellow-500" : "bg-red-500"
          )}
        />
      </div>
    </div>
  );
}
