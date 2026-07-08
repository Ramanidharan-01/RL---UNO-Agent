import React from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuthStore } from './stores/authStore';
import {
  Home, Zap, Eye, Users, BarChart3, Trophy,
  User, Settings, Info, Play, Menu, X,
} from 'lucide-react';

const navItems = [
  { path: '/', icon: Home, label: 'Home' },
  { path: '/dashboard', icon: Play, label: 'Dashboard' },
  { path: '/play', icon: Zap, label: 'Play vs AI' },
  { path: '/simulation', icon: Eye, label: 'Simulation' },
  { path: '/multiplayer', icon: Users, label: 'Multiplayer' },
  { path: '/stats', icon: BarChart3, label: 'Stats' },
  { path: '/leaderboard', icon: Trophy, label: 'Leaderboard' },
];

const secondaryNav = [
  { path: '/profile', icon: User, label: 'Profile' },
  { path: '/settings', icon: Settings, label: 'Settings' },
  { path: '/about', icon: Info, label: 'About' },
];

export const App: React.FC = () => {
  const location = useLocation();
  const { isAuthenticated, user } = useAuthStore();
  const [mobileMenuOpen, setMobileMenuOpen] = React.useState(false);

  return (
    <div className="min-h-screen flex">
      {/* ── Sidebar (desktop) ──────────────────────────────────────────── */}
      <aside className="hidden lg:flex flex-col w-64 bg-surface-900/80 border-r border-surface-800/50 backdrop-blur-sm">
        {/* Logo */}
        <Link to="/" className="p-6 flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-uno-red via-uno-yellow to-uno-blue flex items-center justify-center text-white font-display font-bold text-lg">
            U
          </div>
          <span className="font-display font-bold text-xl bg-gradient-to-r from-white to-surface-300 bg-clip-text text-transparent">
            UNO Arena
          </span>
        </Link>

        {/* Main nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all text-sm ${
                  isActive
                    ? 'bg-accent/15 text-accent font-medium'
                    : 'text-surface-400 hover:text-white hover:bg-surface-800/50'
                }`}
              >
                <item.icon className="w-5 h-5" />
                {item.label}
                {isActive && (
                  <motion.div
                    layoutId="nav-indicator"
                    className="absolute left-0 w-1 h-6 bg-accent rounded-r-full"
                  />
                )}
              </Link>
            );
          })}

          <div className="border-t border-surface-800/50 my-4" />

          {secondaryNav.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all text-sm ${
                  isActive
                    ? 'bg-accent/15 text-accent font-medium'
                    : 'text-surface-400 hover:text-white hover:bg-surface-800/50'
                }`}
              >
                <item.icon className="w-5 h-5" />
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* User badge */}
        <div className="p-4 border-t border-surface-800/50">
          {isAuthenticated && user ? (
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full bg-gradient-to-br from-accent to-purple-600 flex items-center justify-center font-bold text-sm">
                {user.username[0].toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium truncate">{user.username}</div>
                <div className="text-xs text-surface-500">ELO {user.elo_rating.toFixed(0)}</div>
              </div>
            </div>
          ) : (
            <Link to="/settings" className="btn-secondary w-full text-center text-sm py-2">
              Sign In
            </Link>
          )}
        </div>
      </aside>

      {/* ── Mobile header ──────────────────────────────────────────────── */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-50 bg-surface-900/95 backdrop-blur-sm border-b border-surface-800/50">
        <div className="flex items-center justify-between px-4 py-3">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-uno-red via-uno-yellow to-uno-blue flex items-center justify-center text-white font-bold text-sm">U</div>
            <span className="font-display font-bold">UNO Arena</span>
          </Link>
          <button onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {mobileMenuOpen && (
          <motion.nav
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            className="px-4 pb-4 space-y-1"
          >
            {[...navItems, ...secondaryNav].map((item) => (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm text-surface-400 hover:text-white hover:bg-surface-800/50"
              >
                <item.icon className="w-5 h-5" />
                {item.label}
              </Link>
            ))}
          </motion.nav>
        )}
      </div>

      {/* ── Main content ───────────────────────────────────────────────── */}
      <main className="flex-1 lg:pt-0 pt-14 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
};
