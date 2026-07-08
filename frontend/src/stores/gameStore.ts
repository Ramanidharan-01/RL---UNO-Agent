import { create } from 'zustand';
import type { GameState, MoveEvent, GameStatus, MatchMode } from '@/types/game';
import { apiClient } from '@/api/client';

interface GameStore {
  // State
  gameState: GameState | null;
  status: GameStatus;
  matchId: string | null;
  events: MoveEvent[];
  error: string | null;

  // Actions
  createGame: (mode?: MatchMode, seed?: number) => Promise<void>;
  submitAction: (actionIdx: number) => Promise<void>;
  fetchState: (matchId: string) => Promise<void>;
  forfeit: () => Promise<void>;
  reset: () => void;
  setGameState: (state: GameState) => void;
  addEvents: (events: MoveEvent[]) => void;
}

export const useGameStore = create<GameStore>((set, get) => ({
  gameState: null,
  status: 'idle',
  matchId: null,
  events: [],
  error: null,

  createGame: async (mode: MatchMode = 'human_vs_ai', seed?: number) => {
    set({ status: 'loading', error: null });
    try {
      const { data } = await apiClient.post('/api/game/create', { mode, seed });
      set({
        gameState: data,
        matchId: data.match_id,
        status: 'playing',
        events: [],
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to create game';
      set({ status: 'error', error: message });
    }
  },

  submitAction: async (actionIdx: number) => {
    const { matchId } = get();
    if (!matchId) return;

    set({ status: 'waiting' });
    try {
      const { data } = await apiClient.post(`/api/game/${matchId}/action`, {
        action_idx: actionIdx,
      });

      const newEvents = data.events || [];
      set((state) => ({
        gameState: data,
        status: data.done ? 'done' : 'playing',
        events: [...state.events, ...newEvents],
      }));
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to submit action';
      set({ status: 'error', error: message });
    }
  },

  fetchState: async (matchId: string) => {
    try {
      const { data } = await apiClient.get(`/api/game/${matchId}`);
      set({ gameState: data, matchId, status: data.done ? 'done' : 'playing' });
    } catch {
      set({ status: 'error', error: 'Match not found' });
    }
  },

  forfeit: async () => {
    const { matchId } = get();
    if (!matchId) return;
    try {
      await apiClient.delete(`/api/game/${matchId}`);
    } finally {
      set({ gameState: null, matchId: null, status: 'idle', events: [] });
    }
  },

  reset: () => {
    set({ gameState: null, matchId: null, status: 'idle', events: [], error: null });
  },

  setGameState: (state: GameState) => {
    set({ gameState: state, status: state.done ? 'done' : 'playing' });
  },

  addEvents: (events: MoveEvent[]) => {
    set((state) => ({ events: [...state.events, ...events] }));
  },
}));
