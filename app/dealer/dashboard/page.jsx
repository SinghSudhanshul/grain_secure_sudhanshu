'use client';
export const dynamic = 'force-dynamic';


import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { signOut } from 'next-auth/react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Shield, Package, TrendingUp, CheckCircle, LogOut } from 'lucide-react';

export default function DealerDashboard() {
    const { data: session, status } = useSession();
    const router = useRouter();
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (status === 'unauthenticated') {
            router.push('/login');
        } else if (session?.user?.role !== 'DEALER') {
            router.push('/');
        } else {
            fetchDashboard();
        }
    }, [session, status]);

    const fetchDashboard = async () => {
        try {
            const res = await fetch('/api/dealer/dashboard');
            const data = await res.json();
            setStats(data);
            setLoading(false);
        } catch (error) {
            console.error('Failed to fetch dashboard:', error);
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <header className="bg-white border-b shadow-sm">
                <div className="container mx-auto px-4 py-4 flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <Shield className="h-8 w-8 text-blue-600" />
                        <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                            GrainSecure
                        </span>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="text-right">
                            <p className="text-sm font-medium">{session?.user?.name}</p>
                            <p className="text-xs text-gray-600">Fair Price Shop Dealer</p>
                        </div>
                        <Button variant="outline" onClick={() => signOut({ callbackUrl: '/' })}>
                            <LogOut className="h-4 w-4 mr-2" />
                            Logout
                        </Button>
                    </div>
                </div>
            </header>

            <div className="container mx-auto px-4 py-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold mb-2">Dealer Dashboard</h1>
                    <p className="text-gray-600">Stock management and beneficiary verification</p>
                </div>

                {/* Stats */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-sm text-gray-600 flex items-center gap-2">
                                <Package className="h-4 w-4" />
                                Rice Stock
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">{stats?.stock?.rice || 0} kg</div>
                        </CardContent>
                    </Card>
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-sm text-gray-600 flex items-center gap-2">
                                <Package className="h-4 w-4" />
                                Wheat Stock
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">{stats?.stock?.wheat || 0} kg</div>
                        </CardContent>
                    </Card>
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-sm text-gray-600 flex items-center gap-2">
                                <TrendingUp className="h-4 w-4" />
                                Today's Distribution
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-blue-600">{stats?.todayCount || 0}</div>
                        </CardContent>
                    </Card>
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-sm text-gray-600 flex items-center gap-2">
                                <CheckCircle className="h-4 w-4" />
                                Compliance Score
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-green-600">{stats?.complianceScore || 95}%</div>
                        </CardContent>
                    </Card>
                </div>

                {/* Quick Actions */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className="hover:shadow-lg transition-shadow cursor-pointer">
                        <CardHeader>
                            <CardTitle>📱 Verify Beneficiary</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <p className="text-gray-600 mb-4">
                                Search by ration card ID and verify beneficiary identity using OTP or face match
                            </p>
                            <Button className="w-full">Start Verification</Button>
                        </CardContent>
                    </Card>

                    <Card className="hover:shadow-lg transition-shadow cursor-pointer">
                        <CardHeader>
                            <CardTitle>📦 Update Stock</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <p className="text-gray-600 mb-4">
                                Log new stock deliveries and update outgoing distribution records
                            </p>
                            <Button className="w-full" variant="outline">Manage Stock</Button>
                        </CardContent>
                    </Card>
                </div>

                {/* Info Card */}
                <Card className="mt-6 bg-blue-50 border-blue-200">
                    <CardContent className="pt-6">
                        <p className="text-sm text-blue-900">
                            <strong>💡 Tip:</strong> All verification attempts are logged in the tamper-proof audit trail.
                            Use manual override only when absolutely necessary and provide detailed reasons.
                        </p>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
