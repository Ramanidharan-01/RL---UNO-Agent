// ─── Game State Types ────────────────────────────────────────────────────────

export interface Card {
  card_idx: number;
  name: string;
}

export interface Action {
  action_idx: number;
  name: string;
}

export interface Opponent {
  seat: number;
  hand_size: number;
  is_current: boolean;
  uno: boolean;
}

export interface ColorInfo {
  idx: number;
  name: string;
}

export interface MoveEvent {
  player: number;
  player_type: 'human' | 'ai' | 'random' | 'greedy' | 'agent';
  action_idx: number;
  action_name: string;
  done: boolean;
  winner: number;
  value_estimate?: number;
  top_actions?: TopAction[];
}

export interface TopAction {
  action_idx: number;
  action_name: string;
  prob: number;
}

export interface GameState {
  match_id: string;
  mode: string;
  step_count: number;
  current_player: number;
  hero_seat: number;
  viewing_player: number;
  is_your_turn: boolean;
  direction: 'clockwise' | 'counter-clockwise';
  done: boolean;
  winner: number;
  winner_name: string | null;
  top_card: Card;
  current_color: ColorInfo;
  last_action: Action;
  deck_size: number;
  your_hand: Card[];
  your_hand_size: number;
  opponents: Record<string, Opponent>;
  legal_actions: Action[];
  wild_actions: Action[];
  wd4_actions: Action[];
  events?: MoveEvent[];
  event?: MoveEvent;
}

export type MatchMode = 'human_vs_ai' | 'agent_vs_random' | 'agent_vs_greedy';

export type GameStatus = 'idle' | 'loading' | 'playing' | 'waiting' | 'done' | 'error';

// ─── Simulation Types ────────────────────────────────────────────────────────

export interface SimulationControl {
  paused: boolean;
  speed: number;
}

// ─── Replay Types ────────────────────────────────────────────────────────────

export interface ReplayData {
  match_id: string;
  mode: string;
  seed: number | null;
  status: string;
  winner_seat: number | null;
  total_turns: number | null;
  started_at: string | null;
  ended_at: string | null;
  duration_seconds: number | null;
  turns: ReplayTurn[];
}

export interface ReplayTurn {
  step: number;
  player_seat: number;
  player_type: string;
  action_idx: number;
  action_name: string;
  ai_value_estimate: number | null;
  ai_top_actions: TopAction[] | null;
}
