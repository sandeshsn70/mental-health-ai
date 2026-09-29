// src/utils/api.js
// Axios-based API client for Flask backend

import axios from 'axios';

const API_BASE =
  process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Attach user_id to requests that contain a JSON body
api.interceptors.request.use(config => {
  const userId = localStorage.getItem('user_id') || 'anonymous';

  if (config.data && typeof config.data === 'object') {
    config.data.user_id = userId;
  }

  return config;
});

export const predictMood = (text) =>
  api.post('/predict', { text }).then(response => response.data);

export const chatMessage = (message, label) =>
  api.post('/chat', { message, label }).then(response => response.data);

export const getHistory = (limit = 50) =>
  api.get('/history', {
    params: { limit }
  }).then(response => response.data);

export const getMetrics = () =>
  api.get('/metrics').then(response => response.data);

export const getFeatures = (label) =>
  api.get('/features', {
    params: { label }
  }).then(response => response.data);

export const getHelplines = () =>
  api.get('/helplines').then(response => response.data);

export default api;
