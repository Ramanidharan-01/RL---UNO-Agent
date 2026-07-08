import { create } from 'zustand';
import type { GameState, MoveEvent, SimulationControl } from '@/types/game';
import { apiClient } from '@/api/client';

interface SimulationStore {
  // State
  gameState: GameState | null;
  matchId: string | null;
  control: SimulationControl;
  events: MoveEvent[];
  isRunning: boolean;
  isDone: boolean;
  ws: WebSocket | null;

  // Actions
  createSimulation: (mode?: string, seed?: number) => Promise<void>;
  connectWebSocket: (matchId: string) => void;
  disconnect: () => void;
  pause: () => void;
  resume: () => void;
  step: () => void;
  setSpeed: (speed: number) => void;
  stop: () => void;
  reset: () => void;
}

export const useSimulationStore = create<SimulationStore>((set, get) => ({
  gameState: null,
  matchId: null,
  control: { paused: false, speed: 1.0 },
  events: [],
  isRunning: false,
  isDone: false,
  ws: null,

  createSimulation: async (mode = 'agent_vs_random', seed?: number) => {
    try {
      const { data } = await apiClient.post('/api/simulation/create', { mode, seed });
      set({
        gameState: data,
        matchId: data.match_id,
        events: [],
        isDone: false,
        control: { paused: false, speed: 1.0 },
      });
    } catch (err) {
      console.error('Failed to create simulation:', err);
    }
  },

  connectWebSocket: (matchId: string) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/simulation/${matchId}/stream`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      set({ isRunning: true });
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);

      switch (msg.type) {
        case 'state':
          set({ gameState: msg.data });
          break;
        case 'step':
          set((state) => ({
            gameState: msg.data,
            events: [...state.events, msg.data.event].filter(Boolean),
          }));
          break;
        case 'done':
          set({ isDone: true, isRunning: false });
          break;
        case 'control':
          set((state) => ({
            control: {
              ...state.control,
              ...(msg.paused !== undefined ? { paused: msg.paused } : {}),
              ...(msg.speed !== undefined ? { speed: msg.speed } : {}),
            },
          }));
          break;
        case 'error':
          console.error('Simulation error:', msg.message);
          break;
      }
    };

    ws.onclose = () => {
      set({ isRunning: false });
    };

    set({ ws });
  },

  disconnect: () => {
    const { ws } = get();
    if (ws) {
      ws.close();
      set({ ws: null, isRunning: false });
    }
  },

  pause: () => {
    const { ws } = get();
    ws?.send(JSON.stringify({ action: 'pause' }));
    set((state) => ({ control: { ...state.control, paused: true } }));
  },

  resume: () => {
    const { ws } = get();
    ws?.send(JSON.stringify({ action: 'resume' }));
    set((state) => ({ control: { ...state.control, paused: false } }));
  },

  step: () => {
    const { ws } = get();
    ws?.send(JSON.stringify({ action: 'step' }));
  },

  setSpeed: (speed: number) => {
    const { ws } = get();
    ws?.send(JSON.stringify({ action: 'speed', value: speed }));
    set((state) => ({ control: { ...state.control, speed } }));
  },

  stop: () => {
    const { ws } = get();
    ws?.send(JSON.stringify({ action: 'stop' }));
  },

  reset: () => {
    get().disconnect();
    set({
      gameState: null,
      matchId: null,
      control: { paused: false, speed: 1.0 },
      events: [],
      isRunning: false,
      isDone: false,
    });
  },
}));
