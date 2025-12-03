import { create } from 'zustand';
import { TelemetryState } from '../types/store';

export const useTelemetryStore = create<TelemetryState>((set) => ({
    logs: [],
    isConnected: false,
    eps: 0,

    addLog: (newLog) => set((state) => ({
        // Keep only last 1000 logs in memory to save RAM
        logs: [...state.logs.slice(-999), newLog]
    })),

    setConnection: (status) => set({ isConnected: status }),
    clearBuffer: () => set({ logs: [] })
}));
