import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { useSimulationStore } from '@/stores/simulationStore';
import { Play, Pause, SkipForward, Square, Gauge } from 'lucide-react';

const speedOptions = [0.5, 1, 2, 5, 10];

export const SimulationPage: React.FC = () => {
  const {
    gameState, matchId, control, events, isRunning, isDone,
    createSimulation, connectWebSocket, disconnect,
    pause, resume, step, setSpeed, stop, reset,
  } = useSimulationStore();

  const handleStart = async (mode: string) => {
    reset();
    await createSimulation(mode);
  };

  // Connect WebSocket when matchId is set
  useEffect(() => {
    if (matchId && !isRunning && !isDone) {
      connectWebSocket(matchId);
    }
    return () => { disconnect(); };
  }, [matchId]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="page-container">
      <h1 className="section-title">AI Simulation</h1>

      {/* ── Mode Selection ────────────────────────────────────────────── */}
      {!matchId && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-2xl mx-auto">
          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => handleStart('agent_vs_random')}
            className="glass-card-hover p-8 text-center"
          >
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-accent to-purple-600 flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">🎲</span>
            </div>
            <h3 className="text-lg font-semibold mb-2">Agent vs Random</h3>
            <p className="text-sm text-surface-400">
              Your trained RL agent plays against 3 random opponents.
            </p>
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => handleStart('agent_vs_greedy')}
            className="glass-card-hover p-8 text-center"
          >
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-uno-blue to-cyan-600 flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">🧠</span>
            </div>
            <h3 className="text-lg font-semibold mb-2">Agent vs Greedy</h3>
            <p className="text-sm text-surface-400">
              Your trained RL agent plays against 3 heuristic greedy bots.
            </p>
          </motion.button>
        </div>
      )}

      {/* ── Simulation Viewer ─────────────────────────────────────────── */}
      {gameState && (
        <div className="space-y-6">
          {/* Controls */}
          <div className="glass-card p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              {control.paused ? (
                <motion.button whileTap={{ scale: 0.9 }} onClick={resume} className="btn-primary px-4 py-2 flex items-center gap-2">
                  <Play className="w-4 h-4" /> Resume
                </motion.button>
              ) : (
                <motion.button whileTap={{ scale: 0.9 }} onClick={pause} className="btn-secondary px-4 py-2 flex items-center gap-2">
                  <Pause className="w-4 h-4" /> Pause
                </motion.button>
              )}
              <motion.button whileTap={{ scale: 0.9 }} onClick={step} className="btn-secondary px-4 py-2 flex items-center gap-2">
                <SkipForward className="w-4 h-4" /> Step
              </motion.button>
              <motion.button whileTap={{ scale: 0.9 }} onClick={stop} className="btn-danger px-4 py-2 flex items-center gap-2">
                <Square className="w-4 h-4" /> Stop
              </motion.button>
            </div>

            {/* Speed control */}
            <div className="flex items-center gap-3">
              <Gauge className="w-4 h-4 text-surface-400" />
              <span className="text-sm text-surface-400">Speed:</span>
              {speedOptions.map((s) => (
                <button
                  key={s}
                  onClick={() => setSpeed(s)}
                  className={`px-3 py-1 rounded-lg text-sm font-mono transition-colors ${
                    control.speed === s
                      ? 'bg-accent text-white'
                      : 'bg-surface-700/60 text-surface-400 hover:bg-surface-600/60'
                  }`}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>

          {/* Game state display */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main view */}
            <div className="lg:col-span-2 glass-card p-6">
              <div className="grid grid-cols-4 gap-4 mb-6">
                {[0, 1, 2, 3].map((seat) => {
                  const isAgent = seat === 0;
                  const handSize = seat === gameState.viewing_player
                    ? gameState.your_hand_size
                    : gameState.opponents[String(seat)]?.hand_size ?? 0;
                  const isCurrent = gameState.current_player === seat;

                  return (
                    <motion.div
                      key={seat}
                      className={`glass-card p-4 text-center ${isCurrent ? 'ring-2 ring-accent' : ''}`}
                      animate={isCurrent ? { scale: [1, 1.02, 1] } : {}}
                      transition={{ repeat: Infinity, duration: 2 }}
                    >
                      <div className="text-xs text-surface-500 mb-1">
                        {isAgent ? '🤖 Agent' : `Player ${seat}`}
                      </div>
                      <div className="text-2xl font-bold mb-1">{handSize}</div>
                      <div className="text-xs text-surface-500">cards</div>
                      {handSize === 1 && <div className="badge badge-danger mt-1 text-xs">UNO!</div>}
                    </motion.div>
                  );
                })}
              </div>

              {/* Current state */}
              <div className="flex items-center justify-center gap-8">
                <div className="text-center">
                  <div className="text-xs text-surface-500 mb-1">Top Card</div>
                  <div className="text-lg font-bold">{gameState.top_card.name}</div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-surface-500 mb-1">Color</div>
                  <div className="text-lg font-bold capitalize">{gameState.current_color.name}</div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-surface-500 mb-1">Step</div>
                  <div className="text-lg font-bold">{gameState.step_count}</div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-surface-500 mb-1">Deck</div>
                  <div className="text-lg font-bold">{gameState.deck_size}</div>
                </div>
              </div>

              {isDone && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="text-center mt-6"
                >
                  <div className="text-2xl font-display font-bold text-accent">
                    {gameState.winner === 0 ? '🤖 Agent Wins!' : `Player ${gameState.winner} Wins!`}
                  </div>
                  <button onClick={reset} className="btn-primary mt-4">New Simulation</button>
                </motion.div>
              )}
            </div>

            {/* Event log */}
            <div className="glass-card p-4 max-h-[500px] overflow-y-auto">
              <h3 className="text-sm font-semibold text-surface-300 mb-3 sticky top-0 bg-surface-800/80 py-1">
                Move Log ({events.length})
              </h3>
              <div className="space-y-1">
                {events.slice(-50).map((event, i) => (
                  <div key={i} className="text-xs text-surface-400 flex items-center gap-2 py-1 border-b border-surface-800/50">
                    <span className={`w-1.5 h-1.5 rounded-full ${
                      event.player_type === 'agent' ? 'bg-accent' :
                      event.player_type === 'random' ? 'bg-surface-500' : 'bg-uno-green'
                    }`} />
                    <span className="text-surface-500">P{event.player}</span>
                    <span className="font-mono">{event.action_name}</span>
                    {event.value_estimate != null && (
                      <span className="text-accent text-[0.65rem]">v={event.value_estimate.toFixed(2)}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
