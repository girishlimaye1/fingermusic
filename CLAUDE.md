# CLAUDE.md — FingerMusic

## Project Overview

FingerMusic is a browser-based web app that uses TensorFlow.js MoveNet pose detection to map body poses to solfège notes (Do, Re, Mi, Fa, So, La, Ti) for interactive music training. Users capture custom poses during a setup phase, then perform those poses in real-time to trigger musical notes.

## Tech Stack

- **Framework**: Next.js 13.3.1 (Pages Router)
- **Language**: TypeScript (strict mode disabled)
- **UI**: React 18.2, Tailwind CSS 3.3.1
- **ML/Pose Detection**: TensorFlow.js 4.4 with WebGL backend, @tensorflow-models/pose-detection (MoveNet SINGLEPOSE_LIGHTNING)
- **Audio**: Web Audio API (oscillator-based tone generation as fallback), optional MP3 files in `public/notes/`
- **HTTP**: axios 1.3.6 (included as dependency, not currently used in source)
- **Deployment**: Vercel via GitHub Actions
- **Node version**: 18 (specified in CI)

## Repository Structure

```
fingermusic/
├── pages/
│   ├── _app.tsx          # App wrapper, imports global CSS
│   ├── _document.tsx     # Custom HTML document (meta tags, theme-color)
│   └── index.tsx         # Main application (all app logic lives here)
├── styles/
│   └── globals.css       # Tailwind directives + mobile-optimized base styles
├── public/
│   └── notes/
│       └── README.md     # Instructions for adding MP3 note files
├── .github/
│   └── workflows/
│       └── deploy-vercel.yml  # CI/CD: build + deploy to Vercel on push to main
├── next.config.js        # Next.js config (webpack fallback for fs/net/tls)
├── tailwind.config.js    # Tailwind content paths: pages/ and components/
├── postcss.config.js     # PostCSS: tailwindcss + autoprefixer
├── tsconfig.json         # TypeScript config (strict: false, jsx: preserve)
├── package.json          # Dependencies and scripts
└── .gitignore            # Ignores node_modules, .next, out, .env*, .DS_Store
```

## Commands

```bash
npm install       # Install dependencies
npm run dev       # Start development server (http://localhost:3000)
npm run build     # Production build
npm run start     # Start production server
npm run lint      # Run Next.js linter (ESLint)
```

There is no test framework configured. There are no unit or integration tests.

## Architecture

This is a single-page application — all logic resides in `pages/index.tsx`. The app has two screens managed by a `screen` state variable:

1. **Setup screen** (`screen === 'setup'`): User selects a solfège note, strikes a pose in front of the camera, and captures it. Captured poses are stored in a `poseLibrary` state object keyed by note name.

2. **Play screen** (`screen === 'play'`): Real-time pose detection runs in an animation frame loop, comparing the current pose against saved poses. When a match is found (Euclidean distance < 0.35 threshold), the corresponding note is played.

### Key Algorithms

- **Pose normalization** (`normalizePose`): Centers keypoints around their centroid and scales by max distance from center. Only keypoints with confidence > 0.3 are used.
- **Pose matching** (`matchPose`): Compares normalized keypoints using average Euclidean distance across valid keypoint pairs. Match threshold is 0.35.
- **Note playback** (`playNote`): Tries to play an MP3 from `/notes/{note}.mp3`, falls back to Web Audio API oscillator generating a sine wave at the correct frequency. Debounced at 500ms per note.

### Rendering

- The camera feed is mirrored horizontally via `transform: -scale-x-100` on both the video and overlay canvas.
- Pose skeleton is drawn on a canvas overlay with green (#10b981) keypoints and connections.
- Keypoints with confidence score below 0.3 are filtered out from rendering.

## Code Conventions

- **Single-file architecture**: All application logic is in `pages/index.tsx`. There is no `components/` directory yet (though Tailwind config includes it).
- **Functional components**: React functional components with hooks (`useState`, `useRef`, `useEffect`).
- **Type definitions**: Inline types at the top of `index.tsx` (`SavedPose`, `PoseLibrary`). No separate types file.
- **Styling**: Tailwind utility classes exclusively; no CSS modules or styled-components. Mobile-first design with gradient backgrounds and rounded corners.
- **Constants**: `NOTES` array defined at module scope (`['do', 're', 'mi', 'fa', 'so', 'la', 'ti']`).
- **No ESLint config file**: Uses Next.js built-in ESLint via `next lint`.
- **No Prettier config**: No explicit formatting configuration.

## Deployment

- **Platform**: Vercel
- **CI trigger**: Push to `main` branch or manual workflow dispatch
- **Required GitHub secrets**: `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`, `VERCEL_TOKEN`
- **Workflow**: `.github/workflows/deploy-vercel.yml` — installs deps, builds, deploys via Vercel CLI

## Development Notes

- The app requires camera access (`navigator.mediaDevices.getUserMedia`) and works best on devices with a front-facing camera.
- WebGL backend is required for TensorFlow.js pose detection.
- The webpack config in `next.config.js` provides fallbacks for Node.js modules (`fs`, `net`, `tls`) that TensorFlow.js may reference.
- MP3 audio files are optional — the app generates tones programmatically when files are missing.
- The `axios` dependency is included but not currently used in the source code.
