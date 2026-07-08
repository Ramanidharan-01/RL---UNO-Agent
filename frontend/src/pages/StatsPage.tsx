import React from 'react';
import { motion } from 'framer-motion';
import { BarChart3, TrendingUp, Zap, Target, Award, Clock } from 'lucide-react';

export const StatsPage: React.FC = () => {
  return (
    <div className="page-container">
      <h1 className="section-title">Your Statistics</h1>

      {/* ── Stat Cards ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
        {[
          { icon: Zap, label: 'Games', value: '—', color: 'text-accent' },
          { icon: TrendingUp, label: 'Win Rate', value: '—', color: 'text-uno-green' },
          { icon: Target, label: 'ELO', value: '1000', color: 'text-uno-blue' },
          { icon: Award, label: 'Streak', value: '—', color: 'text-uno-yellow' },
          { icon: Clock, label: 'Avg Length', value: '—', color: 'text-uno-red' },
          { icon: BarChart3, label: 'Cards Played', value: '—', color: 'text-purple-400' },
        ].map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08 }}
            className="glass-card p-4 text-center"
          >
            <stat.icon className={`w-5 h-5 mx-auto mb-2 ${stat.color}`} />
            <div className="text-2xl font-bold">{stat.value}</div>
            <div className="text-xs text-surface-500">{stat.label}</div>
          </motion.div>
        ))}
      </div>

      {/* ── Match History ─────────────────────────────────────────────── */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold mb-4">Match History</h2>
        <div className="text-center py-12 text-surface-500">
          <BarChart3 className="w-10 h-10 mx-auto mb-3 opacity-40" />
          <p>Play some games to see your history here.</p>
        </div>
      </div>
    </div>
  );
};
