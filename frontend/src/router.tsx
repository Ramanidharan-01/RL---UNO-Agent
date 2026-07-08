import React, { Suspense, lazy } from 'react';
import { createBrowserRouter } from 'react-router-dom';
import { App } from './App';

// Lazy-loaded pages for code splitting
const LandingPage = lazy(() => import('./pages/LandingPage').then(m => ({ default: m.LandingPage })));
const DashboardPage = lazy(() => import('./pages/DashboardPage').then(m => ({ default: m.DashboardPage })));
const PlayAIPage = lazy(() => import('./pages/PlayAIPage').then(m => ({ default: m.PlayAIPage })));
const SimulationPage = lazy(() => import('./pages/SimulationPage').then(m => ({ default: m.SimulationPage })));
const MultiplayerLobbyPage = lazy(() => import('./pages/MultiplayerLobbyPage').then(m => ({ default: m.MultiplayerLobbyPage })));
const ReplayPage = lazy(() => import('./pages/ReplayPage').then(m => ({ default: m.ReplayPage })));
const StatsPage = lazy(() => import('./pages/StatsPage').then(m => ({ default: m.StatsPage })));
const LeaderboardPage = lazy(() => import('./pages/LeaderboardPage').then(m => ({ default: m.LeaderboardPage })));
const ProfilePage = lazy(() => import('./pages/ProfilePage').then(m => ({ default: m.ProfilePage })));
const SettingsPage = lazy(() => import('./pages/SettingsPage').then(m => ({ default: m.SettingsPage })));
const AboutPage = lazy(() => import('./pages/AboutPage').then(m => ({ default: m.AboutPage })));

const PageLoader = () => (
  <div className="flex items-center justify-center min-h-[60vh]">
    <div className="flex gap-2">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="w-3 h-3 rounded-full bg-accent animate-bounce"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  </div>
);

export const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <Suspense fallback={<PageLoader />}><LandingPage /></Suspense> },
      { path: 'dashboard', element: <Suspense fallback={<PageLoader />}><DashboardPage /></Suspense> },
      { path: 'play', element: <Suspense fallback={<PageLoader />}><PlayAIPage /></Suspense> },
      { path: 'simulation', element: <Suspense fallback={<PageLoader />}><SimulationPage /></Suspense> },
      { path: 'multiplayer', element: <Suspense fallback={<PageLoader />}><MultiplayerLobbyPage /></Suspense> },
      { path: 'replay', element: <Suspense fallback={<PageLoader />}><ReplayPage /></Suspense> },
      { path: 'replay/:matchId', element: <Suspense fallback={<PageLoader />}><ReplayPage /></Suspense> },
      { path: 'stats', element: <Suspense fallback={<PageLoader />}><StatsPage /></Suspense> },
      { path: 'leaderboard', element: <Suspense fallback={<PageLoader />}><LeaderboardPage /></Suspense> },
      { path: 'profile', element: <Suspense fallback={<PageLoader />}><ProfilePage /></Suspense> },
      { path: 'settings', element: <Suspense fallback={<PageLoader />}><SettingsPage /></Suspense> },
      { path: 'about', element: <Suspense fallback={<PageLoader />}><AboutPage /></Suspense> },
    ],
  },
]);
