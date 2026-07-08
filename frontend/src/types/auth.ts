export interface User {
  id: string;
  email: string;
  username: string;
  avatar_url: string | null;
  elo_rating: number;
  is_verified: boolean;
  created_at: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface PlayerStats {
  user_id: string;
  username: string;
  elo_rating: number;
  games_played: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  avg_game_length: number;
  total_cards_played: number;
  total_draw_actions: number;
  wild_cards_played: number;
  current_streak: number;
  best_streak: number;
  fastest_win_turns: number | null;
}

export interface LeaderboardEntry {
  rank: number;
  user_id: string;
  username: string;
  avatar_url: string | null;
  elo: number;
  games_played: number;
  wins: number;
  win_rate: number;
}

export interface MatchHistoryEntry {
  match_id: string;
  mode: string;
  status: string;
  winner_seat: number | null;
  total_turns: number | null;
  started_at: string | null;
  ended_at: string | null;
  duration_seconds: number | null;
}
