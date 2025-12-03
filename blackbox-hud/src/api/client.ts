import axios, { AxiosInstance, InternalAxiosRequestConfig } from 'axios';

// 1. Get the API URL from the environment, or default to localhost
// This enables the "Remote Access" capability we discussed.
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

// 2. Create the Axios Instance
const client: AxiosInstance = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
    timeout: 10000, // 10 seconds timeout
});

// 3. Request Interceptor (The "Middleware")
// Automatically adds the JWT Token to every request if the user is logged in.
client.interceptors.request.use(
    (config: InternalAxiosRequestConfig) => {
        // Read token from LocalStorage (or Zustand store)
        const token = localStorage.getItem('blackbox_token');
        
        if (token && config.headers) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// 4. Response Interceptor (Error Handling)
// If the server says "401 Unauthorized", force the user to logout.
client.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response && error.response.status === 401) {
            // Token expired or invalid
            localStorage.removeItem('blackbox_token');
            window.location.href = '/login'; // Redirect to login page
        }
        return Promise.reject(error);
    }
);

export default client;