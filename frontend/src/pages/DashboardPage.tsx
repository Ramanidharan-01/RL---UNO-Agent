import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';
import { Zap, Eye, Users, BarChart3, Trophy, Clock } from 'lucide-react';

const quickActions = [
  { icon: Zap, label: 'Play vs AI', path: '/play', color: 'from-accent to-purple-600' },
  { icon: Eye, label: 'Watch Sim', path: '/simulation', color: 'from-uno-blue to-cyan-600' },
  { icon: Users, label: 'Multiplayer', path: '/multiplayer', color: 'from-uno-green to-emerald-600' },
  { icon: Trophy, label: 'Leaderboard', path: '/leaderboard', color: 'from-uno-yellow to-orange-500' },
];

export const DashboardPage: React.FC = () => {
  const { user, isAuthenticated } = useAuthStore();

  return (
    <div className="page-container">
      {/* ── Welcome ──────────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-display font-bold">
          {isAuthenticated ? `Welcome back, ${user?.username}` : 'Welcome to UNO Arena'}
        </h1>
        <p className="text-surface-400 mt-2">Choose your next game mode</p>
      </motion.div>

      {/* ── Quick Actions ────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {quickActions.map((action, i) => (
          <motion.div
            key={action.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
          >
            <Link to={action.path}>
              <div className="glass-card-hover p-6 text-center group">
                <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${action.color} flex items-center justify-center mx-auto mb-3 group-hover:scale-110 transition-transform`}>
                  <action.icon className="w-7 h-7 text-white" />
                </div>
                <span className="font-semibold">{action.label}</span>
              </div>
            </Link>
          </motion.div>
        ))}
      </div>

      {/* ── Stats Summary ────────────────────────────────────────────── */}
      {isAuthenticated && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="glass-card p-6">
            <div className="flex items-center gap-3 mb-2">
              <BarChart3 className="w-5 h-5 text-accent" />
              <span className="text-sm text-surface-400">ELO Rating</span>
            </div>
            <div className="text-3xl font-bold">{user?.elo_rating?.toFixed(0) ?? '1000'}</div>
          </div>
          <div className="glass-card p-6">
            <div className="flex items-center gap-3 mb-2">
              <Trophy className="w-5 h-5 text-uno-yellow" />
              <span className="text-sm text-surface-400">Win Rate</span>
            </div>
            <div className="text-3xl font-bold">—</div>
            <Link to="/stats" className="text-sm text-accent hover:underline">View stats →</Link>
          </div>
          <div className="glass-card p-6">
            <div className="flex items-center gap-3 mb-2">
              <Clock className="w-5 h-5 text-uno-green" />
              <span className="text-sm text-surface-400">Recent Games</span>
            </div>
            <div className="text-3xl font-bold">—</div>
            <Link to="/stats" className="text-sm text-accent hover:underline">View history →</Link>
          </div>
        </div>
      )}

      {/* ── Login prompt for guests ──────────────────────────────────── */}
      {!isAuthenticated && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="glass-card p-8 text-center"
        >
          <h3 className="text-lg font-semibold mb-2">Track your progress</h3>
          <p className="text-surface-400 mb-4">Create an account to save stats, compete on the leaderboard, and play multiplayer.</p>
          <Link to="/settings" className="btn-primary inline-block">Sign Up</Link>
        </motion.div>
      )}
    </div>
  );
};
