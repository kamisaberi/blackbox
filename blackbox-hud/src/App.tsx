import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from './pages/Dashboard';

// Simple Login Placeholder
const Login = () => (
    <div className="h-screen w-screen bg-black flex items-center justify-center text-white">
        <div className="w-96 p-8 border border-white/10 rounded bg-neutral-900 text-center">
            <h1 className="text-2xl font-bold mb-4">BLACKBOX</h1>
            <input type="text" placeholder="Username" className="w-full mb-4 p-2 bg-black border border-white/20 rounded" />
            <input type="password" placeholder="Password" className="w-full mb-6 p-2 bg-black border border-white/20 rounded" />
            <button
                onClick={() => window.location.href = '/dashboard'}
                className="w-full bg-blue-600 hover:bg-blue-700 p-2 rounded font-bold"
            >
                INITIALIZE
            </button>
        </div>
    </div>
);

function App() {
  return (
    <BrowserRouter>
        <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
        </Routes>
    </BrowserRouter>
  );
}

export default App;