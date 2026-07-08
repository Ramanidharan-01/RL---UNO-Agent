import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { useGameStore } from '@/stores/gameStore';
import { GameBoard } from '@/components/game/GameBoard';
import { Loader2, RefreshCw, Flag } from 'lucide-react';

export const PlayAIPage: React.FC = () => {
  const { gameState, status, createGame, submitAction, forfeit, reset } = useGameStore();

  const handleNewGame = () => {
    reset();
    createGame('human_vs_ai');
  };

  const handleDrawCard = () => {
    submitAction(60); // DRAW_ACTION = 60
  };

  // Auto-create game on mount if no active game
  useEffect(() => {
    if (status === 'idle') {
      createGame('human_vs_ai');
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="page-container">
      {/* ── Header ───────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="section-title mb-0">Play vs AI</h1>
        <div className="flex gap-3">
          {status === 'playing' && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={forfeit}
              className="btn-danger flex items-center gap-2 px-4 py-2 text-sm"
            >
              <Flag className="w-4 h-4" />
              Forfeit
            </motion.button>
          )}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleNewGame}
            className="btn-secondary flex items-center gap-2 px-4 py-2 text-sm"
          >
            <RefreshCw className="w-4 h-4" />
            New Game
          </motion.button>
        </div>
      </div>

      {/* ── Loading ──────────────────────────────────────────────────── */}
      {status === 'loading' && (
        <div className="flex flex-col items-center justify-center h-[600px] glass-card">
          <Loader2 className="w-12 h-12 text-accent animate-spin mb-4" />
          <p className="text-surface-400">Setting up the game...</p>
        </div>
      )}

      {/* ── Error ────────────────────────────────────────────────────── */}
      {status === 'error' && (
        <div className="flex flex-col items-center justify-center h-[600px] glass-card">
          <p className="text-danger text-lg mb-4">Something went wrong</p>
          <button onClick={handleNewGame} className="btn-primary">Try Again</button>
        </div>
      )}

      {/* ── Game Board ───────────────────────────────────────────────── */}
      {gameState && (status === 'playing' || status === 'waiting' || status === 'done') && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card overflow-hidden"
        >
          <GameBoard
            gameState={gameState}
            onPlayCard={submitAction}
            onDrawCard={handleDrawCard}
            disabled={status === 'waiting'}
          />
        </motion.div>
      )}

      {/* ── AI Thinking Indicator ────────────────────────────────────── */}
      {status === 'waiting' && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed bottom-8 right-8 glass-card p-4 flex items-center gap-3"
        >
          <div className="flex gap-1">
            {[0, 1, 2].map((i) => (
              <motion.div
                key={i}
                className="w-2 h-2 rounded-full bg-accent"
                animate={{ y: [0, -8, 0] }}
                transition={{ repeat: Infinity, delay: i * 0.15, duration: 0.6 }}
              />
            ))}
          </div>
          <span className="text-sm text-surface-400">AI is thinking...</span>
        </motion.div>
      )}

      {/* ── Game Over Actions ────────────────────────────────────────── */}
      {status === 'done' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex justify-center mt-6 gap-4"
        >
          <button onClick={handleNewGame} className="btn-primary flex items-center gap-2">
            <RefreshCw className="w-4 h-4" />
            Play Again
          </button>
        </motion.div>
      )}
    </div>
  );
};
