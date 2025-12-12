import React from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { useTelemetryStore } from '../../store/useTelemetryStore';

// Helper to generate mock historical data for the chart init
const generateInitialData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
        time: i,
        safe: Math.floor(Math.random() * 500) + 1000,
        threat: Math.floor(Math.random() * 50),
    }));
};

export const VelocityChart = () => {
    // In a real app, this data would come from the Store history or API stats endpoint
    // For MVP, we use the store's stats object to drive a single dynamic point
    const stats = useTelemetryStore((state) => state.stats);

    // We maintain local chart state to create the "scrolling" effect
    const [data, setData] = React.useState(generateInitialData());

    React.useEffect(() => {
        const interval = setInterval(() => {
            setData(currentData => {
                const newData = [...currentData.slice(1)]; // Remove oldest
                newData.push({
                    time: new Date().getSeconds(),
                    safe: stats.eps || Math.random() * 100, // Fallback if EPS is 0
                    threat: stats.threat_count % 100 // Mock variation based on threat count
                });
                return newData;
            });
        }, 1000);

        return () => clearInterval(interval);
    }, [stats]);

    return (
        <div className="h-48 bg-black/40 border border-white/10 rounded-lg p-4 backdrop-blur-sm">
            <h3 className="text-xs font-bold text-gray-400 mb-2 uppercase tracking-wider">Network Velocity (EPS)</h3>
            <div className="h-full w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data}>
                        <defs>
                            <linearGradient id="colorSafe" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#10B981" stopOpacity={0.3}/>
                                <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
                            </linearGradient>
                            <linearGradient id="colorThreat" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3}/>
                                <stop offset="95%" stopColor="#EF4444" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                        <XAxis dataKey="time" hide />
                        <YAxis hide domain={[0, 'auto']} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#000', borderColor: '#333' }}
                            itemStyle={{ fontSize: '12px' }}
                        />
                        <Area
                            type="monotone"
                            dataKey="safe"
                            stroke="#10B981"
                            fillOpacity={1}
                            fill="url(#colorSafe)"
                            isAnimationActive={false} // Performance optimization
                        />
                        <Area
                            type="monotone"
                            dataKey="threat"
                            stroke="#EF4444"
                            fillOpacity={1}
                            fill="url(#colorThreat)"
                            isAnimationActive={false}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};