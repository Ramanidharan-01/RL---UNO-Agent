import React from 'react';
import { motion } from 'framer-motion';
import { Cpu, Layers, Zap, Database, Globe, Shield } from 'lucide-react';

export const AboutPage: React.FC = () => {
  return (
    <div className="page-container max-w-4xl mx-auto">
      <h1 className="section-title">About UNO Arena</h1>

      <div className="space-y-6">
        {/* ── Overview ────────────────────────────────────────────────── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-card p-8">
          <h2 className="text-xl font-display font-bold mb-4 flex items-center gap-2">
            <Cpu className="w-6 h-6 text-accent" /> The AI
          </h2>
          <p className="text-surface-300 leading-relaxed mb-4">
            UNO Arena features a reinforcement learning agent trained with <strong className="text-white">Proximal Policy Optimization (PPO)</strong> using
            a <strong className="text-white">Gated Transformer-XL (GTrXL)</strong> architecture for recurrent memory. The agent was trained in a
            self-play regime with fictitious self-play, competing against historical snapshots and average policy opponents.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Architecture', value: 'GTrXL' },
              { label: 'Hidden Dim', value: '192' },
              { label: 'Memory Len', value: '12' },
              { label: 'Action Space', value: '61' },
            ].map((item) => (
              <div key={item.label} className="bg-surface-800/60 rounded-xl p-3 text-center">
                <div className="text-xs text-surface-500">{item.label}</div>
                <div className="font-mono font-bold text-accent">{item.value}</div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* ── Architecture ────────────────────────────────────────────── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-card p-8">
          <h2 className="text-xl font-display font-bold mb-4 flex items-center gap-2">
            <Layers className="w-6 h-6 text-uno-blue" /> Architecture
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { icon: Globe, title: 'Frontend', items: ['React + TypeScript', 'Tailwind CSS', 'Framer Motion', 'Zustand State'] },
              { icon: Zap, title: 'Backend', items: ['FastAPI + WebSocket', 'JAX/Flax Inference', 'Redis Game State', 'PostgreSQL History'] },
              { icon: Shield, title: 'Infrastructure', items: ['Docker Compose', 'Nginx Reverse Proxy', 'CI/CD Pipeline', 'JWT Auth'] },
            ].map((col) => (
              <div key={col.title} className="bg-surface-800/40 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <col.icon className="w-5 h-5 text-accent" />
                  <span className="font-semibold">{col.title}</span>
                </div>
                <ul className="space-y-1">
                  {col.items.map((item) => (
                    <li key={item} className="text-sm text-surface-400 flex items-center gap-2">
                      <div className="w-1 h-1 rounded-full bg-accent" />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </motion.div>

        {/* ── Game Engine ─────────────────────────────────────────────── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass-card p-8">
          <h2 className="text-xl font-display font-bold mb-4 flex items-center gap-2">
            <Database className="w-6 h-6 text-uno-green" /> Game Engine
          </h2>
          <p className="text-surface-300 leading-relaxed">
            The UNO game engine is implemented entirely in JAX for JIT-compiled, hardware-accelerated performance.
            It enforces the complete UNO ruleset including Draw Two, Reverse, Skip, Wild, Wild Draw Four, and color selection.
            The engine is the single source of truth — clients never decide legal moves. All game state is serialized via
            pickle and stored in Redis for sub-millisecond access.
          </p>
        </motion.div>
      </div>
    </div>
  );
};
