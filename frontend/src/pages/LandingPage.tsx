import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Zap, Eye, Users, BarChart3, Trophy, Cpu, ArrowRight } from 'lucide-react';

const features = [
  {
    icon: Zap,
    title: 'Play vs AI',
    description: 'Challenge a trained reinforcement learning agent in real-time UNO.',
    color: 'from-accent to-purple-600',
    link: '/play',
  },
  {
    icon: Eye,
    title: 'Watch Simulations',
    description: 'Observe 4-player AI battles with live stats and confidence scores.',
    color: 'from-uno-blue to-cyan-600',
    link: '/simulation',
  },
  {
    icon: Users,
    title: 'Multiplayer',
    description: 'Play with friends or join public lobbies. AI fills empty seats.',
    color: 'from-uno-green to-emerald-600',
    link: '/multiplayer',
  },
  {
    icon: BarChart3,
    title: 'Statistics',
    description: 'Track your performance with detailed analytics and game history.',
    color: 'from-uno-yellow to-orange-500',
    link: '/stats',
  },
  {
    icon: Trophy,
    title: 'Leaderboard',
    description: 'Compete for the top rank with ELO-based matchmaking.',
    color: 'from-uno-red to-rose-600',
    link: '/leaderboard',
  },
  {
    icon: Cpu,
    title: 'AI Insights',
    description: 'See what the neural network thinks — top actions, confidence, and value estimates.',
    color: 'from-pink-500 to-violet-600',
    link: '/about',
  },
];

const floatingCards = [
  { color: '#E53935', label: '+2', x: '10%', y: '20%', rotate: -15, delay: 0 },
  { color: '#1E88E5', label: '7', x: '85%', y: '15%', rotate: 12, delay: 0.5 },
  { color: '#43A047', label: '⇄', x: '5%', y: '70%', rotate: -25, delay: 1 },
  { color: '#FDD835', label: '⊘', x: '90%', y: '65%', rotate: 20, delay: 1.5 },
  { color: '#6C63FF', label: 'W', x: '50%', y: '10%', rotate: 5, delay: 0.8 },
];

export const LandingPage: React.FC = () => {
  return (
    <div className="relative overflow-hidden">
      {/* ── Floating background cards ─────────────────────────────────── */}
      {floatingCards.map((card, i) => (
        <motion.div
          key={i}
          className="absolute w-16 h-24 rounded-xl flex items-center justify-center text-white font-bold text-xl shadow-lg opacity-10 pointer-events-none"
          style={{
            background: `linear-gradient(135deg, ${card.color}, ${card.color}88)`,
            left: card.x,
            top: card.y,
          }}
          animate={{
            y: [0, -20, 0],
            rotate: [card.rotate, card.rotate + 5, card.rotate],
          }}
          transition={{
            repeat: Infinity,
            duration: 6,
            delay: card.delay,
            ease: 'easeInOut',
          }}
        >
          {card.label}
        </motion.div>
      ))}

      {/* ── Hero Section ──────────────────────────────────────────────── */}
      <section className="relative page-container min-h-[80vh] flex flex-col items-center justify-center text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h1 className="text-6xl md:text-8xl font-display font-extrabold mb-6">
            <span className="bg-gradient-to-r from-uno-red via-uno-yellow via-uno-green to-uno-blue bg-clip-text text-transparent">
              UNO Arena
            </span>
          </h1>
          <p className="text-xl md:text-2xl text-surface-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            Play UNO against a <span className="text-accent font-semibold">trained AI agent</span>.
            Watch simulations. Compete on the leaderboard.
            Powered by deep reinforcement learning.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/play">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="btn-primary text-lg px-8 py-4 flex items-center gap-2"
              >
                <Zap className="w-5 h-5" />
                Play Now
                <ArrowRight className="w-5 h-5" />
              </motion.button>
            </Link>
            <Link to="/simulation">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="btn-secondary text-lg px-8 py-4 flex items-center gap-2"
              >
                <Eye className="w-5 h-5" />
                Watch AI Play
              </motion.button>
            </Link>
          </div>
        </motion.div>
      </section>

      {/* ── Features Grid ─────────────────────────────────────────────── */}
      <section className="page-container pb-20">
        <motion.h2
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="section-title text-center text-3xl mb-12"
        >
          Everything you need
        </motion.h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
            >
              <Link to={feature.link}>
                <div className="glass-card-hover p-6 h-full group">
                  <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                    <feature.icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
                  <p className="text-surface-400 text-sm leading-relaxed">{feature.description}</p>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── Tech Stack Banner ──────────────────────────────────────────── */}
      <section className="page-container pb-20">
        <div className="glass-card p-8 text-center">
          <h3 className="text-lg font-display font-semibold text-surface-300 mb-4">Built with</h3>
          <div className="flex flex-wrap justify-center gap-6 text-surface-500">
            {['JAX/Flax', 'PPO + GTrXL', 'FastAPI', 'PostgreSQL', 'Redis', 'React', 'TypeScript', 'WebSocket'].map((tech) => (
              <span key={tech} className="px-4 py-2 bg-surface-800/60 rounded-lg text-sm font-mono">{tech}</span>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};
