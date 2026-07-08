import React from 'react';
import { motion } from 'framer-motion';
import { User, Award, Clock, Target } from 'lucide-react';
import { useAuthStore } from '@/stores/authStore';

export const ProfilePage: React.FC = () => {
  const { user, isAuthenticated } = useAuthStore();

  if (!isAuthenticated || !user) {
    return (
      <div className="page-container text-center py-20">
        <User className="w-16 h-16 mx-auto text-surface-600 mb-4" />
        <h2 className="text-xl font-semibold mb-2">Not logged in</h2>
        <p className="text-surface-500">Sign in to view your profile.</p>
      </div>
    );
  }

  return (
    <div className="page-container max-w-3xl mx-auto">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        {/* ── Profile Header ──────────────────────────────────────────── */}
        <div className="glass-card p-8 text-center mb-8">
          <div className="w-24 h-24 rounded-full bg-gradient-to-br from-accent to-purple-600 flex items-center justify-center mx-auto mb-4 text-4xl font-display font-bold">
            {user.username[0].toUpperCase()}
          </div>
          <h1 className="text-2xl font-display font-bold">{user.username}</h1>
          <p className="text-surface-400">{user.email}</p>
          <div className="flex justify-center gap-6 mt-4">
            <div className="text-center">
              <Target className="w-5 h-5 text-accent mx-auto mb-1" />
              <div className="font-bold">{user.elo_rating.toFixed(0)}</div>
              <div className="text-xs text-surface-500">ELO</div>
            </div>
            <div className="text-center">
              <Award className="w-5 h-5 text-uno-yellow mx-auto mb-1" />
              <div className="font-bold">—</div>
              <div className="text-xs text-surface-500">Rank</div>
            </div>
            <div className="text-center">
              <Clock className="w-5 h-5 text-uno-green mx-auto mb-1" />
              <div className="font-bold">{new Date(user.created_at).toLocaleDateString()}</div>
              <div className="text-xs text-surface-500">Joined</div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};
