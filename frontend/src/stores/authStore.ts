import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User, TokenPair } from '@/types/auth';
import { apiClient } from '@/api/client';

interface AuthState {
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  login: (email: string, password: string) => Promise<void>;
  register: (email: string, username: string, password: string) => Promise<void>;
  logout: () => void;
  setTokens: (tokens: TokenPair) => void;
  setUser: (user: User) => void;
  fetchProfile: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      accessToken: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: false,

      login: async (email: string, password: string) => {
        set({ isLoading: true });
        try {
          const { data } = await apiClient.post<TokenPair>('/api/auth/login', {
            email,
            password,
          });
          set({
            accessToken: data.access_token,
            refreshToken: data.refresh_token,
            isAuthenticated: true,
          });
          // Fetch user profile
          await get().fetchProfile();
        } finally {
          set({ isLoading: false });
        }
      },

      register: async (email: string, username: string, password: string) => {
        set({ isLoading: true });
        try {
          const { data } = await apiClient.post<TokenPair>('/api/auth/register', {
            email,
            username,
            password,
          });
          set({
            accessToken: data.access_token,
            refreshToken: data.refresh_token,
            isAuthenticated: true,
          });
          await get().fetchProfile();
        } finally {
          set({ isLoading: false });
        }
      },

      logout: () => {
        // Fire-and-forget server logout
        const token = get().accessToken;
        if (token) {
          apiClient.post('/api/auth/logout').catch(() => {});
        }
        set({
          user: null,
          accessToken: null,
          refreshToken: null,
          isAuthenticated: false,
        });
      },

      setTokens: (tokens: TokenPair) => {
        set({
          accessToken: tokens.access_token,
          refreshToken: tokens.refresh_token,
          isAuthenticated: true,
        });
      },

      setUser: (user: User) => set({ user }),

      fetchProfile: async () => {
        try {
          const { data } = await apiClient.get<User>('/api/auth/me');
          set({ user: data });
        } catch {
          // If profile fetch fails, clear auth state
          set({
            user: null,
            accessToken: null,
            refreshToken: null,
            isAuthenticated: false,
          });
        }
      },
    }),
    {
      name: 'uno-arena-auth',
      partialize: (state) => ({
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
