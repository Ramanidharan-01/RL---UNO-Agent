import React from 'react';
import ReactDOM from 'react-dom/client';
import { RouterProvider } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { router } from './router';
import './styles/globals.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
    <Toaster
      position="bottom-right"
      toastOptions={{
        style: {
          background: '#1a202c',
          color: '#e2e8f0',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: '0.75rem',
        },
      }}
    />
  </React.StrictMode>
);
