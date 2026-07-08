import React from 'react';
import { motion } from 'framer-motion';
import { Users, Plus, ArrowRight } from 'lucide-react';

export const MultiplayerLobbyPage: React.FC = () => {
  return (
    <div className="page-container">
      <h1 className="section-title">Multiplayer</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* ── Create Lobby ────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card p-6"
        >
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Plus className="w-5 h-5 text-accent" /> Create Lobby
          </h2>
          <div className="space-y-4">
            <div>
              <label className="text-sm text-surface-400 block mb-1">Lobby Name</label>
              <input type="text" placeholder="My UNO Room" className="input-field" />
            </div>
            <div>
              <label className="text-sm text-surface-400 block mb-1">Max Players</label>
              <select className="input-field">
                <option value="2">2 Players</option>
                <option value="3">3 Players</option>
                <option value="4" selected>4 Players</option>
              </select>
            </div>
            <label className="flex items-center gap-2 text-sm text-surface-300">
              <input type="checkbox" defaultChecked className="rounded" />
              Fill empty seats with AI
            </label>
            <button className="btn-primary w-full flex items-center justify-center gap-2">
              <Plus className="w-4 h-4" /> Create
            </button>
          </div>
        </motion.div>

        {/* ── Open Lobbies ────────────────────────────────────────────── */}
        <div className="lg:col-span-2">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-card p-6"
          >
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Users className="w-5 h-5 text-uno-green" /> Open Lobbies
            </h2>

            <div className="space-y-3">
              {/* Placeholder lobbies */}
              {['Chill Game', 'Competitive UNO', 'Beginners Welcome'].map((name, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="flex items-center justify-between p-4 bg-surface-800/40 rounded-xl hover:bg-surface-800/60 transition-colors"
                >
                  <div>
                    <div className="font-medium">{name}</div>
                    <div className="text-sm text-surface-500">{i + 1}/4 players</div>
                  </div>
                  <button className="btn-secondary px-4 py-2 text-sm flex items-center gap-1">
                    Join <ArrowRight className="w-3 h-3" />
                  </button>
                </motion.div>
              ))}

              {/* Empty state */}
              <div className="text-center py-8 text-surface-500">
                <Users className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No open lobbies right now. Create one!</p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};
