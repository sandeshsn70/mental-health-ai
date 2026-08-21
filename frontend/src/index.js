import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Generate anonymous user ID for session persistence
if (!localStorage.getItem('user_id')) {
  localStorage.setItem('user_id', 'user_' + Math.random().toString(36).substr(2, 9));
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
