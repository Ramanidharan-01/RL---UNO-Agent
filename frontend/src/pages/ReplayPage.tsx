import React from 'react';
import { motion } from 'framer-motion';
import { Play, SkipBack, SkipForward, Clock } from 'lucide-react';

export const ReplayPage: React.FC = () => {
  return (
    <div className="page-container">
      <h1 className="section-title">Match Replay</h1>

      <div className="glass-card p-8 text-center">
        <Clock className="w-16 h-16 mx-auto text-surface-600 mb-4" />
        <h2 className="text-xl font-semibold mb-2">No replay selected</h2>
        <p className="text-surface-400 mb-6">
          Select a match from your history or enter a match ID to watch the replay.
        </p>

        <div className="max-w-md mx-auto">
          <input type="text" placeholder="Enter match ID" className="input-field mb-4" />
          <button className="btn-primary w-full flex items-center justify-center gap-2">
            <Play className="w-4 h-4" /> Load Replay
          </button>
        </div>

        {/* Playback controls placeholder */}
        <div className="mt-8 flex items-center justify-center gap-4 opacity-30">
          <button className="btn-secondary p-2"><SkipBack className="w-5 h-5" /></button>
          <button className="btn-primary p-3"><Play className="w-6 h-6" /></button>
          <button className="btn-secondary p-2"><SkipForward className="w-5 h-5" /></button>
        </div>
      </div>
    </div>
  );
};
