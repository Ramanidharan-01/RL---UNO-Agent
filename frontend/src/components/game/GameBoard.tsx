import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { GameState } from '@/types/game';
import { UnoCard } from './UnoCard';
import { PlayerHand } from './PlayerHand';
import { Zap, RotateCcw, RotateCw, Layers, AlertTriangle } from 'lucide-react';

interface GameBoardProps {
  gameState: GameState;
  onPlayCard: (actionIdx: number) => void;
  onDrawCard: () => void;
  disabled?: boolean;
}

const colorBgMap: Record<string, string> = {
  red: 'from-uno-red/20 to-transparent',
  yellow: 'from-uno-yellow/20 to-transparent',
  green: 'from-uno-green/20 to-transparent',
  blue: 'from-uno-blue/20 to-transparent',
};

export const GameBoard: React.FC<GameBoardProps> = ({
  gameState,
  onPlayCard,
  onDrawCard,
  disabled = false,
}) => {
  const [showColorPicker, setShowColorPicker] = useState(false);
  const [pendingWildType, setPendingWildType] = useState<'wild' | 'wd4' | null>(null);

  const handlePlayCard = (actionIdx: number) => {
    // Check if this is a wild/WD4 action that needs color selection
    if (actionIdx >= 52 && actionIdx <= 55) {
      setPendingWildType('wild');
      setShowColorPicker(true);
      return;
    }
    if (actionIdx >= 56 && actionIdx <= 59) {
      setPendingWildType('wd4');
      setShowColorPicker(true);
      return;
    }
    onPlayCard(actionIdx);
  };

  const handleColorSelect = (colorIdx: number) => {
    if (pendingWildType === 'wild') {
      onPlayCard(52 + colorIdx);
    } else if (pendingWildType === 'wd4') {
      onPlayCard(56 + colorIdx);
    }
    setShowColorPicker(false);
    setPendingWildType(null);
  };

  const opponents = Object.values(gameState.opponents).sort((a, b) => a.seat - b.seat);
  const colorGradient = colorBgMap[gameState.current_color.name] || '';

  return (
    <div className={`relative w-full h-full min-h-[600px] bg-gradient-to-b ${colorGradient} rounded-3xl p-6 flex flex-col`}>

      {/* ── Opponents (top row) ──────────────────────────────────────── */}
      <div className="flex justify-around items-start mb-8">
        {opponents.map((opp) => (
          <motion.div
            key={opp.seat}
            className={`glass-card p-4 min-w-[120px] text-center ${
              opp.is_current ? 'ring-2 ring-accent animate-pulse-glow' : ''
            }`}
            animate={opp.is_current ? { scale: [1, 1.02, 1] } : {}}
            transition={{ repeat: Infinity, duration: 2 }}
          >
            <div className="text-sm text-surface-400 mb-1">Player {opp.seat}</div>
            <div className="flex justify-center gap-0.5 mb-2">
              {Array.from({ length: Math.min(opp.hand_size, 10) }).map((_, i) => (
                <div key={i} className="w-3 h-5 bg-surface-600 rounded-sm" />
              ))}
              {opp.hand_size > 10 && (
                <span className="text-xs text-surface-500 ml-1">+{opp.hand_size - 10}</span>
              )}
            </div>
            <div className="text-lg font-bold">{opp.hand_size} cards</div>
            {opp.uno && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="badge badge-danger mt-1"
              >
                UNO!
              </motion.div>
            )}
          </motion.div>
        ))}
      </div>

      {/* ── Center area (discard + draw pile) ─────────────────────────── */}
      <div className="flex-1 flex items-center justify-center gap-12">
        {/* Discard pile */}
        <div className="text-center">
          <div className="text-xs text-surface-500 mb-2 uppercase tracking-wider">Discard</div>
          <motion.div
            key={gameState.top_card.card_idx}
            initial={{ scale: 0.5, rotate: -30 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <UnoCard card={gameState.top_card} size="lg" />
          </motion.div>
          <div className="mt-2 flex items-center justify-center gap-2">
            <div
              className="w-4 h-4 rounded-full border-2 border-white/30"
              style={{
                backgroundColor: {
                  red: '#E53935', yellow: '#FDD835', green: '#43A047', blue: '#1E88E5',
                }[gameState.current_color.name] || '#666',
              }}
            />
            <span className="text-sm text-surface-400 capitalize">{gameState.current_color.name}</span>
          </div>
        </div>

        {/* Direction indicator */}
        <div className="flex flex-col items-center gap-2">
          <motion.div
            animate={{ rotate: gameState.direction === 'clockwise' ? 360 : -360 }}
            transition={{ repeat: Infinity, duration: 4, ease: 'linear' }}
          >
            {gameState.direction === 'clockwise' ? (
              <RotateCw className="w-8 h-8 text-accent/60" />
            ) : (
              <RotateCcw className="w-8 h-8 text-accent/60" />
            )}
          </motion.div>
          <span className="text-xs text-surface-500">Step {gameState.step_count}</span>
        </div>

        {/* Draw pile */}
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={onDrawCard}
          disabled={disabled || !gameState.is_your_turn}
          className="text-center group"
        >
          <div className="text-xs text-surface-500 mb-2 uppercase tracking-wider">Draw</div>
          <div className="relative">
            <div className="w-28 h-40 rounded-xl bg-gradient-to-br from-surface-700 to-surface-800 border-2 border-surface-600 flex flex-col items-center justify-center shadow-card group-hover:shadow-card-hover transition-all">
              <Layers className="w-8 h-8 text-accent/70 mb-1" />
              <span className="text-lg font-bold">{gameState.deck_size}</span>
              <span className="text-xs text-surface-500">cards</span>
            </div>
          </div>
        </motion.button>
      </div>

      {/* ── Status bar ────────────────────────────────────────────────── */}
      <div className="flex justify-center my-4">
        <AnimatePresence mode="wait">
          {gameState.is_your_turn && !gameState.done && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="badge badge-accent text-base px-4 py-2 flex items-center gap-2"
            >
              <Zap className="w-4 h-4" />
              Your turn — play a card or draw
            </motion.div>
          )}
          {!gameState.is_your_turn && !gameState.done && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="badge badge-warning text-base px-4 py-2"
            >
              Waiting for Player {gameState.current_player}...
            </motion.div>
          )}
          {gameState.done && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className={`text-xl font-display font-bold px-6 py-3 rounded-xl ${
                gameState.winner === gameState.viewing_player
                  ? 'bg-success/20 text-green-400'
                  : 'bg-danger/20 text-red-400'
              }`}
            >
              {gameState.winner === gameState.viewing_player ? '🎉 You Win!' : `${gameState.winner_name} Wins!`}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Player hand (bottom) ──────────────────────────────────────── */}
      <PlayerHand
        cards={gameState.your_hand}
        legalActions={gameState.legal_actions}
        onPlayCard={handlePlayCard}
        disabled={disabled || !gameState.is_your_turn || gameState.done}
      />

      {/* ── Color picker modal ────────────────────────────────────────── */}
      <AnimatePresence>
        {showColorPicker && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 rounded-3xl"
            onClick={() => { setShowColorPicker(false); setPendingWildType(null); }}
          >
            <motion.div
              initial={{ scale: 0.5 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.5 }}
              className="glass-card p-8"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-lg font-display font-bold text-center mb-4">Choose a color</h3>
              <div className="grid grid-cols-2 gap-4">
                {['red', 'yellow', 'green', 'blue'].map((color, idx) => (
                  <motion.button
                    key={color}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => handleColorSelect(idx)}
                    className="w-20 h-20 rounded-xl shadow-lg transition-shadow"
                    style={{
                      backgroundColor: { red: '#E53935', yellow: '#FDD835', green: '#43A047', blue: '#1E88E5' }[color],
                      boxShadow: `0 0 20px ${{ red: 'rgba(229,57,53,0.4)', yellow: 'rgba(253,216,53,0.4)', green: 'rgba(67,160,71,0.4)', blue: 'rgba(30,136,229,0.4)' }[color]}`,
                    }}
                  >
                    <span className="text-white text-2xl font-bold capitalize">{color[0]}</span>
                  </motion.button>
                ))}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
