import React, { useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { useTelemetryStore } from '../store/useTelemetryStore';
import { LogViewer } from '../components/stream/LogViewer';
import { VelocityChart } from '../components/visualizations/VelocityChart';
import { Shield, Activity, AlertTriangle, Database } from 'lucide-react';

const StatCard = ({ label, value, icon: Icon, color }: any) => (
    <div className="bg-black/40 border border-white/10 p-4 rounded-lg flex items-center space-x-4 backdrop-blur-sm">
        <div className={`p-2 rounded-full bg-opacity-20 ${color.bg}`}>
            <Icon className={`w-6 h-6 ${color.text}`} />
        </div>
        <div>
            <p className="text-xs text-gray-500 uppercase">{label}</p>
            <p className="text-2xl font-mono font-bold text-white">{value}</p>
        </div>
    </div>
);

export default function Dashboard() {
    // Initialize WebSocket connection
    useWebSocket();
    const stats = useTelemetryStore((state) => state.stats);
    const isConnected = useTelemetryStore((state) => state.isConnected);

    return (
        <div className="flex h-screen bg-neutral-900 text-white overflow-hidden bg-[url('/grid-pattern.png')]">

            {/* Sidebar (Simplified for MVP) */}
            <div className="w-16 bg-black border-r border-white/10 flex flex-col items-center py-6 space-y-8 z-10">
                <div className="w-10 h-10 bg-blue-600 rounded flex items-center justify-center font-bold text-xl shadow-[0_0_15px_rgba(37,99,235,0.5)]">
                    B
                </div>
                <nav className="flex-1 space-y-6 w-full flex flex-col items-center">
                    <Shield className="w-6 h-6 text-gray-400 hover:text-white cursor-pointer" />
                    <Activity className="w-6 h-6 text-white cursor-pointer" />
                    <Database className="w-6 h-6 text-gray-400 hover:text-white cursor-pointer" />
                </nav>
                <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 shadow-[0_0_10px_#22c55e]' : 'bg-red-500'}`} title={isConnected ? "Online" : "Offline"} />
            </div>

            {/* Main Content */}
            <div className="flex-1 flex flex-col min-w-0">

                {/* Header Stats */}
                <div className="h-24 border-b border-white/10 flex items-center px-8 space-x-6 bg-black/20">
                    <StatCard
                        label="Total Events"
                        value={stats.total_logs.toLocaleString()}
                        icon={Activity}
                        color={{ bg: 'bg-blue-500', text: 'text-blue-500' }}
                    />
                    <StatCard
                        label="Threats Detected"
                        value={stats.threat_count.toLocaleString()}
                        icon={AlertTriangle}
                        color={{ bg: 'bg-red-500', text: 'text-red-500' }}
                    />
                    <StatCard
                        label="Events / Sec"
                        value={stats.eps.toLocaleString()}
                        icon={Database}
                        color={{ bg: 'bg-green-500', text: 'text-green-500' }}
                    />

                    <div className="flex-1" /> {/* Spacer */}

                    <div className="text-right">
                        <h1 className="text-lg font-bold tracking-tight">BLACKBOX ENGINE</h1>
                        <p className="text-xs text-gray-500 font-mono">v0.1.0-alpha // LIVE</p>
                    </div>
                </div>

                {/* Content Grid */}
                <div className="flex-1 p-6 grid grid-cols-4 gap-6 min-h-0">

                    {/* Left Col: Stream (Takes up 3 cols) */}
                    <div className="col-span-3 flex flex-col min-h-0 space-y-4">
                        <LogViewer />
                    </div>

                    {/* Right Col: Visualization & Widgets */}
                    <div className="col-span-1 flex flex-col space-y-6 min-h-0">
                        <VelocityChart />

                        {/* Threat Feed (Simplified Text List) */}
                        <div className="flex-1 bg-black/40 border border-white/10 rounded-lg p-4 backdrop-blur-sm overflow-hidden flex flex-col">
                            <h3 className="text-xs font-bold text-gray-400 mb-4 uppercase tracking-wider">Active Threats</h3>
                            <div className="flex-1 overflow-y-auto space-y-2 pr-2 font-mono text-xs">
                                {stats.threat_count === 0 && (
                                    <p className="text-gray-600 italic">No active threats detected.</p>
                                )}
                                {/* In a real app, render list of recent alerts here */}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}