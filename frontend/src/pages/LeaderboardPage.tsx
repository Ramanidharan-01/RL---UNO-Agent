import React from 'react';
import { motion } from 'framer-motion';
import { Trophy, Medal } from 'lucide-react';

const placeholderEntries = Array.from({ length: 10 }, (_, i) => ({
  rank: i + 1,
  username: `Player${i + 1}`,
  elo: (1500 - i * 35).toFixed(0),
  games: 50 - i * 3,
  winRate: (70 - i * 4).toFixed(1),
}));

export const LeaderboardPage: React.FC = () => {
  return (
    <div className="page-container">
      <h1 className="section-title flex items-center gap-3">
        <Trophy className="w-7 h-7 text-uno-yellow" />
        Leaderboard
      </h1>

      <div className="glass-card overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-surface-700/50 text-surface-400 text-sm">
              <th className="py-4 px-6 text-left">Rank</th>
              <th className="py-4 px-6 text-left">Player</th>
              <th className="py-4 px-6 text-right">ELO</th>
              <th className="py-4 px-6 text-right">Games</th>
              <th className="py-4 px-6 text-right">Win Rate</th>
            </tr>
          </thead>
          <tbody>
            {placeholderEntries.map((entry, i) => (
              <motion.tr
                key={entry.rank}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                className="border-b border-surface-800/50 hover:bg-surface-800/30 transition-colors"
              >
                <td className="py-4 px-6">
                  <div className="flex items-center gap-2">
                    {entry.rank <= 3 ? (
                      <Medal className={`w-5 h-5 ${
                        entry.rank === 1 ? 'text-yellow-400' :
                        entry.rank === 2 ? 'text-gray-400' : 'text-orange-500'
                      }`} />
                    ) : (
                      <span className="text-surface-500 w-5 text-center">{entry.rank}</span>
                    )}
                  </div>
                </td>
                <td className="py-4 px-6">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-accent to-purple-600 flex items-center justify-center text-sm font-bold">
                      {entry.username[0]}
                    </div>
                    <span className="font-medium">{entry.username}</span>
                  </div>
                </td>
                <td className="py-4 px-6 text-right font-mono font-bold text-accent">{entry.elo}</td>
                <td className="py-4 px-6 text-right text-surface-400">{entry.games}</td>
                <td className="py-4 px-6 text-right">
                  <span className="badge badge-success">{entry.winRate}%</span>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
