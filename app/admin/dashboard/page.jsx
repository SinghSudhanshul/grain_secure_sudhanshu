'use client';

import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import AdminLayout from '@/components/layouts/AdminLayout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Users, Store, Activity, AlertTriangle, IndianRupee, Play, Square, Zap } from 'lucide-react';
import { io } from 'socket.io-client';

let socket;

export default function AdminDashboard() {
    const { data: session, status } = useSession();
    const router = useRouter();
    const [stats, setStats] = useState(null);
    const [recentTransactions, setRecentTransactions] = useState([]);
    const [recentAlerts, setRecentAlerts] = useState([]);
    const [simulatorRunning, setSimulatorRunning] = useState(false);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (status === 'unauthenticated') {
            router.push('/login');
        } else if (session?.user?.role !== 'ADMIN') {
            router.push('/');
        } else {
            fetchDashboardData();
            initializeSocket();
        }

        return () => {
            if (socket) socket.disconnect();
        };
    }, [session, status]);

    const initializeSocket = () => {
        fetch('/api/socket');
        socket = io();

        socket.on('transaction', (data) => {
            setRecentTransactions(prev => [data, ...prev.slice(0, 9)]);
        });

        socket.on('alert', (data) => {
            setRecentAlerts(prev => [data, ...prev.slice(0, 9)]);
        });

        socket.on('simulatorStatus', (data) => {
            setSimulatorRunning(data.running);
        });
    };

    const fetchDashboardData = async () => {
        try {
            const res = await fetch('/api/admin/dashboard');
            const data = await res.json();
            setStats(data.stats);
            setRecentTransactions(data.recentTransactions || []);
            setRecentAlerts(data.recentAlerts || []);
            setLoading(false);
        } catch (error) {
            console.error('Failed to fetch dashboard data:', error);
            setLoading(false);
        }
    };

    const controlSimulator = async (action) => {
        try {
            await fetch('/api/simulator/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action }),
            });
        } catch (error) {
            console.error('Simulator control failed:', error);
        }
    };

    const injectFraud = async (scenario) => {
        try {
            await fetch('/api/simulator/inject-fraud', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scenario }),
            });
        } catch (error) {
            console.error('Fraud injection failed:', error);
        }
    };

    if (loading || !stats) {
        return (
            <AdminLayout>
                <div className="flex items-center justify-center h-96">
                    <div className="text-center">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                        <p className="text-gray-600">Loading dashboard...</p>
                    </div>
                </div>
            </AdminLayout>
        );
    }

    const anomalyTrendData = stats.anomalyTrend || [];
    const riskByShopData = stats.riskByShop || [];
    const anomalyDistData = stats.anomalyDistribution || [];

    const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981'];

    return (
        <AdminLayout>
            <div className="space-y-6">
                {/* Header */}
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-bold">Admin Dashboard</h1>
                        <p className="text-gray-600">Real-time PDS monitoring and fraud detection</p>
                    </div>
                    <Badge variant={simulatorRunning ? "default" : "secondary"} className="text-sm px-3 py-1">
                        {simulatorRunning ? '🟢 Simulator Running' : '🔴 Simulator Stopped'}
                    </Badge>
                </div>

                {/* Simulator Controls */}
                <Card className="bg-gradient-to-r from-purple-50 to-blue-50 border-purple-200">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Zap className="h-5 w-5 text-purple-600" />
                            Digital Twin Simulator Controls
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="flex flex-wrap gap-3">
                            <Button
                                onClick={() => controlSimulator('start')}
                                disabled={simulatorRunning}
                                className="bg-green-600 hover:bg-green-700"
                            >
                                <Play className="h-4 w-4 mr-2" />
                                Start Simulation
                            </Button>
                            <Button
                                onClick={() => controlSimulator('stop')}
                                disabled={!simulatorRunning}
                                variant="destructive"
                            >
                                <Square className="h-4 w-4 mr-2" />
                                Stop Simulation
                            </Button>
                            <div className="border-l border-gray-300 mx-2"></div>
                            <Button
                                onClick={() => injectFraud('stock_diversion')}
                                variant="outline"
                                className="border-orange-300 text-orange-700 hover:bg-orange-50"
                            >
                                💣 Inject: Stock Diversion
                            </Button>
                            <Button
                                onClick={() => injectFraud('ghost_beneficiary')}
                                variant="outline"
                                className="border-red-300 text-red-700 hover:bg-red-50"
                            >
                                👻 Inject: Ghost Beneficiary
                            </Button>
                            <Button
                                onClick={() => injectFraud('bulk_spike')}
                                variant="outline"
                                className="border-yellow-300 text-yellow-700 hover:bg-yellow-50"
                            >
                                📈 Inject: Bulk Spike
                            </Button>
                        </div>
                    </CardContent>
                </Card>

                {/* KPI Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <KPICard
                        title="Total Beneficiaries"
                        value={stats.totalBeneficiaries?.toLocaleString() || '0'}
                        icon={Users}
                        color="blue"
                    />
                    <KPICard
                        title="FPS Shops"
                        value={stats.totalShops?.toLocaleString() || '0'}
                        icon={Store}
                        color="green"
                    />
                    <KPICard
                        title="Transactions"
                        value={stats.totalTransactions?.toLocaleString() || '0'}
                        icon={Activity}
                        color="purple"
                    />
                    <KPICard
                        title="Anomalies Detected"
                        value={stats.totalAnomalies?.toLocaleString() || '0'}
                        icon={AlertTriangle}
                        color="red"
                    />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <IndianRupee className="h-5 w-5" />
                                Estimated Leakage Prevented
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-4xl font-bold text-green-600">
                                ₹{(stats.leakagePrevented || 0).toLocaleString()}
                            </div>
                            <p className="text-sm text-gray-600 mt-2">Based on detected fraud patterns</p>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>Open Cases</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-4xl font-bold text-orange-600">
                                {stats.openCases || 0}
                            </div>
                            <p className="text-sm text-gray-600 mt-2">Require inspector attention</p>
                        </CardContent>
                    </Card>
                </div>

                {/* Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <Card>
                        <CardHeader>
                            <CardTitle>Anomalies Over Time</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <ResponsiveContainer width="100%" height={250}>
                                <LineChart data={anomalyTrendData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="date" />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} />
                                </LineChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>Risk by Shop</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <ResponsiveContainer width="100%" height={250}>
                                <BarChart data={riskByShopData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="shop" />
                                    <YAxis />
                                    <Tooltip />
                                    <Bar dataKey="riskScore" fill="#ef4444" />
                                </BarChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </div>

                <Card>
                    <CardHeader>
                        <CardTitle>Anomaly Distribution</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                                <Pie
                                    data={anomalyDistData}
                                    cx="50%"
                                    cy="50%"
                                    labelLine={false}
                                    label={(entry) => `${entry.name}: ${entry.value}`}
                                    outerRadius={100}
                                    fill="#8884d8"
                                    dataKey="value"
                                >
                                    {anomalyDistData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>

                {/* Real-time feeds */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <Card>
                        <CardHeader>
                            <CardTitle>Live Transaction Feed</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-2 max-h-80 overflow-y-auto">
                                {recentTransactions.length === 0 ? (
                                    <p className="text-sm text-gray-500">No recent transactions</p>
                                ) : (
                                    recentTransactions.map((txn, idx) => (
                                        <div key={idx} className="text-sm border-l-2 border-blue-400 pl-3 py-1">
                                            <div className="font-medium">{txn.beneficiaryName}</div>
                                            <div className="text-gray-600 text-xs">
                                                {txn.shopName} • {new Date(txn.dateTime).toLocaleTimeString()}
                                            </div>
                                        </div>
                                    ))
                                )}
                            </div>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>New Alerts</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-2 max-h-80 overflow-y-auto">
                                {recentAlerts.length === 0 ? (
                                    <p className="text-sm text-gray-500">No new alerts</p>
                                ) : (
                                    recentAlerts.map((alert, idx) => (
                                        <div key={idx} className="text-sm border-l-2 border-red-400 pl-3 py-1">
                                            <div className="flex items-center gap-2">
                                                <Badge variant={alert.severity === 'CRITICAL' ? 'destructive' : 'secondary'}>
                                                    {alert.severity}
                                                </Badge>
                                                <span className="font-medium">{alert.title}</span>
                                            </div>
                                            <div className="text-gray-600 text-xs mt-1">{alert.description}</div>
                                        </div>
                                    ))
                                )}
                            </div>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </AdminLayout>
    );
}

function KPICard({ title, value, icon: Icon, color }) {
    const colorClasses = {
        blue: 'bg-blue-50 text-blue-600',
        green: 'bg-green-50 text-green-600',
        purple: 'bg-purple-50 text-purple-600',
        red: 'bg-red-50 text-red-600',
    };

    return (
        <Card>
            <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                    <div>
                        <p className="text-sm text-gray-600">{title}</p>
                        <p className="text-3xl font-bold mt-2">{value}</p>
                    </div>
                    <div className={`p-3 rounded-full ${colorClasses[color]}`}>
                        <Icon className="h-6 w-6" />
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}
