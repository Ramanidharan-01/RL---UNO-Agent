import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Settings, Moon, Sun, LogIn, UserPlus } from 'lucide-react';
import { useAuthStore } from '@/stores/authStore';

export const SettingsPage: React.FC = () => {
  const { isAuthenticated, login, register, logout, isLoading } = useAuthStore();
  const [isDark, setIsDark] = useState(true);
  const [tab, setTab] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    try {
      if (tab === 'login') {
        await login(email, password);
      } else {
        await register(email, username, password);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Authentication failed');
    }
  };

  const toggleTheme = () => {
    setIsDark(!isDark);
    document.documentElement.classList.toggle('dark');
  };

  return (
    <div className="page-container max-w-2xl mx-auto">
      <h1 className="section-title flex items-center gap-2">
        <Settings className="w-6 h-6" /> Settings
      </h1>

      {/* ── Auth Section ──────────────────────────────────────────────── */}
      {!isAuthenticated ? (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card p-8 mb-8">
          <div className="flex gap-4 mb-6">
            <button
              onClick={() => setTab('login')}
              className={`flex-1 py-2 rounded-xl font-medium transition-colors ${
                tab === 'login' ? 'bg-accent text-white' : 'bg-surface-700/60 text-surface-400'
              }`}
            >
              <LogIn className="w-4 h-4 inline mr-2" /> Log In
            </button>
            <button
              onClick={() => setTab('register')}
              className={`flex-1 py-2 rounded-xl font-medium transition-colors ${
                tab === 'register' ? 'bg-accent text-white' : 'bg-surface-700/60 text-surface-400'
              }`}
            >
              <UserPlus className="w-4 h-4 inline mr-2" /> Sign Up
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} className="input-field" required />
            {tab === 'register' && (
              <input type="text" placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} className="input-field" required minLength={3} />
            )}
            <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} className="input-field" required minLength={8} />
            {error && <p className="text-danger text-sm">{error}</p>}
            <button type="submit" disabled={isLoading} className="btn-primary w-full">
              {isLoading ? 'Loading...' : tab === 'login' ? 'Log In' : 'Create Account'}
            </button>
          </form>
        </motion.div>
      ) : (
        <div className="glass-card p-6 mb-8 flex items-center justify-between">
          <span className="text-surface-300">Signed in as <strong className="text-white">{useAuthStore.getState().user?.username}</strong></span>
          <button onClick={logout} className="btn-danger px-4 py-2 text-sm">Log Out</button>
        </div>
      )}

      {/* ── Theme Toggle ──────────────────────────────────────────────── */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold mb-4">Appearance</h2>
        <div className="flex items-center justify-between">
          <span className="text-surface-300">Theme</span>
          <button onClick={toggleTheme} className="btn-secondary px-4 py-2 flex items-center gap-2">
            {isDark ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
            {isDark ? 'Dark' : 'Light'}
          </button>
        </div>
      </div>
    </div>
  );
};
