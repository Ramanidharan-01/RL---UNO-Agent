import axios from 'axios';
import { useAuthStore } from '@/stores/authStore';

export const apiClient = axios.create({
  baseURL: '',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor — attach JWT access token
apiClient.interceptors.request.use((config) => {
  const token = useAuthStore.getState().accessToken;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor — handle 401 with token refresh
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      const refreshToken = useAuthStore.getState().refreshToken;
      if (refreshToken) {
        try {
          const { data } = await axios.post('/api/auth/refresh', {
            refresh_token: refreshToken,
          });

          useAuthStore.getState().setTokens(data);
          originalRequest.headers.Authorization = `Bearer ${data.access_token}`;
          return apiClient(originalRequest);
        } catch {
          // Refresh failed — force logout
          useAuthStore.getState().logout();
        }
      } else {
        useAuthStore.getState().logout();
      }
    }

    return Promise.reject(error);
  }
);
