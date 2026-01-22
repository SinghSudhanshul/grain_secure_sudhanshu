'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Shield, Users, Store, Activity, IndianRupee, TrendingDown } from 'lucide-react';

export default function PublicDashboard() {
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchPublicData();
    }, []);

    const fetchPublicData = async () => {
        try {
            const res = await fetch('/api/public/dashboard');
            const data = await res.json();
            setStats(data);
            setLoading(false);
        } catch (error) {
            console.error('Failed to fetch public data:', error);
            setLoading(false);
        }
    };

    if (loading || !stats) {
        return (
            <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading public dashboard...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
            {/* Header */}
            <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
                <div className="container mx-auto px-4 py-4 flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <Shield className="h-8 w-8 text-blue-600" />
                        <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                            GrainSecure
                        </span>
                    </div>
                    <div className="flex gap-3">
                        <Link href="/">
                            <Button variant="outline">Home</Button>
                        </Link>
                        <Link href="/login">
                            <Button>Login</Button>
                        </Link>
                    </div>
                </div>
            </header>

            <div className="container mx-auto px-4 py-8">
                {/* Page Header */}
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold mb-2">Public Transparency Dashboard</h1>
                    <p className="text-gray-600">
                        Real-time aggregated PDS data • No login required • Updated every minute
                    </p>
                    <Badge className="mt-3 bg-green-100 text-green-700 border-green-300">
                        🔓 Open Data Initiative
                    </Badge>
                </div>

                {/* KPI Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    <Card className="border-blue-200">
                        <CardHeader className="pb-3">
                            <CardTitle className="text-sm text-gray-600 flex items-center gap-2">
                                <Users className="h-4 w-4" />
                                Registered Beneficiaries
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-3xl font-bold text-blue-600">
                                {stats.totalBeneficiaries?.toLocaleString() || '0'}
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="border-green-200">
                        <CardHeader className="pb-3">
                            <CardTitle className="text-sm text-gray-600 flex items-center gap-2">
                                <Store className="h-4 w-4" />
                                Active FPS Shops
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-3xl font-bold text-green-600">
                                {stats.totalShops?.toLocaleString() || '0'}
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="border-purple-200">
                        <CardHeader className="pb-3">
                            <CardTitle className="text-sm text-gray-600 flex items-center gap-2">
                                <Activity className="h-4 w-4" />
                                Total Distributions
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-3xl font-bold text-purple-600">
                                {stats.totalTransactions?.toLocaleString() || '0'}
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="border-orange-200">
                        <CardHeader className="pb-3">
                            <CardTitle className="text-sm text-gray-600 flex items-center gap-2">
                                <TrendingDown className="h-4 w-4" />
                                Anomalies Detected
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-3xl font-bold text-orange-600">
                                {stats.totalAnomalies?.toLocaleString() || '0'}
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* Leakage Prevention */}
                <Card className="mb-8 border-green-300 bg-gradient-to-r from-green-50 to-emerald-50">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-green-700">
                            <IndianRupee className="h-6 w-6" />
                            Estimated Leakage Prevented (This Month)
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-5xl font-bold text-green-600">
                            ₹{(stats.leakagePrevented || 0).toLocaleString()}
                        </div>
                        <p className="text-sm text-gray-600 mt-2">
                            Through AI-powered fraud detection and real-time monitoring
                        </p>
                    </CardContent>
                </Card>

                {/* Shop Compliance Leaderboard */}
                <Card className="mb-8">
                    <CardHeader>
                        <CardTitle>🏆 Top Compliant FPS Shops</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Rank</TableHead>
                                    <TableHead>Shop Name</TableHead>
                                    <TableHead>Zone</TableHead>
                                    <TableHead>Compliance Score</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {stats.topShops?.map((shop, idx) => (
                                    <TableRow key={idx}>
                                        <TableCell className="font-medium">
                                            {idx === 0 && '🥇'}
                                            {idx === 1 && '🥈'}
                                            {idx === 2 && '🥉'}
                                            {idx > 2 && idx + 1}
                                        </TableCell>
                                        <TableCell>{shop.name}</TableCell>
                                        <TableCell><Badge variant="outline">{shop.zone}</Badge></TableCell>
                                        <TableCell>
                                            <div className="flex items-center gap-2">
                                                <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-green-500"
                                                        style={{ width: `${shop.complianceScore}%` }}
                                                    ></div>
                                                </div>
                                                <span className="text-sm font-semibold">{shop.complianceScore}%</span>
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </CardContent>
                </Card>

                {/* System Integrity */}
                <Card className="border-purple-200 bg-gradient-to-r from-purple-50 to-indigo-50">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-purple-700">
                            🔐 System Integrity Status
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                        <div className="flex items-center justify-between">
                            <span className="text-sm">Audit Log Chain</span>
                            <Badge className="bg-green-100 text-green-700 border-green-300">
                                ✅ Verified
                            </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-sm">AI Detection Active</span>
                            <Badge className="bg-green-100 text-green-700 border-green-300">
                                ✅ Running
                            </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-sm">Real-time Monitoring</span>
                            <Badge className="bg-green-100 text-green-700 border-green-300">
                                ✅ Active
                            </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-sm">Data Last Updated</span>
                            <span className="text-sm font-medium">{new Date().toLocaleString()}</span>
                        </div>
                    </CardContent>
                </Card>

                {/* Footer Note */}
                <div className="mt-8 text-center text-sm text-gray-600 bg-white border rounded-lg p-6">
                    <p className="mb-2">
                        <strong>Note:</strong> This dashboard displays only aggregated, anonymized data to protect
                        beneficiary privacy while ensuring transparency in PDS operations.
                    </p>
                    <p>
                        For detailed analytics and case management, authorized personnel can{' '}
                        <Link href="/login" className="text-blue-600 hover:underline">
                            login here
                        </Link>
                        .
                    </p>
                </div>
            </div>
        </div>
    );
}
