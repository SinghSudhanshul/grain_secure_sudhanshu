'use client';
export const dynamic = 'force-dynamic';


import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { signOut } from 'next-auth/react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Shield, User, Package, History, AlertCircle, LogOut } from 'lucide-react';

export default function BeneficiaryPortal() {
    const { data: session, status } = useSession();
    const router = useRouter();
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (status === 'unauthenticated') {
            router.push('/login');
        } else if (session?.user?.role !== 'BENEFICIARY') {
            router.push('/');
        } else {
            fetchData();
        }
    }, [session, status]);

    const fetchData = async () => {
        try {
            const res = await fetch('/api/beneficiary/portal');
            const data = await res.json();
            setData(data);
            setLoading(false);
        } catch (error) {
            console.error('Failed to fetch data:', error);
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
        <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
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
                            <p className="text-xs text-gray-600">Beneficiary</p>
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
                    <h1 className="text-3xl font-bold mb-2">My PDS Portal</h1>
                    <p className="text-gray-600">View your entitlements and transaction history</p>
                </div>

                {/* Beneficiary Info */}
                <Card className="mb-6 border-blue-200 bg-gradient-to-r from-blue-50 to-indigo-50">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <User className="h-5 w-5" />
                            My Details
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="grid md:grid-cols-2 gap-4">
                        <div>
                            <p className="text-sm text-gray-600">Ration Card ID</p>
                            <p className="font-semibold">{data?.beneficiary?.rationCardId || 'N/A'}</p>
                        </div>
                        <div>
                            <p className="text-sm text-gray-600">Family Size</p>
                            <p className="font-semibold">{data?.beneficiary?.familySize || 0} members</p>
                        </div>
                        <div>
                            <p className="text-sm text-gray-600">Aadhaar (Masked)</p>
                            <p className="font-semibold">{data?.beneficiary?.aadhaarMasked || 'N/A'}</p>
                        </div>
                        <div>
                            <p className="text-sm text-gray-600">Address</p>
                            <p className="font-semibold">{data?.beneficiary?.address || 'N/A'}</p>
                        </div>
                    </CardContent>
                </Card>

                {/* Current Month Entitlement */}
                <Card className="mb-6 border-green-200">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Package className="h-5 w-5 text-green-600" />
                            Current Month Entitlement
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        {data?.entitlement ? (
                            <div className="grid md:grid-cols-3 gap-6">
                                <div className="text-center p-4 bg-orange-50 rounded-lg">
                                    <div className="text-3xl font-bold text-orange-600">{data.entitlement.riceKg} kg</div>
                                    <p className="text-sm text-gray-600 mt-1">Rice</p>
                                </div>
                                <div className="text-center p-4 bg-amber-50 rounded-lg">
                                    <div className="text-3xl font-bold text-amber-600">{data.entitlement.wheatKg} kg</div>
                                    <p className="text-sm text-gray-600 mt-1">Wheat</p>
                                </div>
                                <div className="text-center p-4 bg-yellow-50 rounded-lg">
                                    <div className="text-3xl font-bold text-yellow-600">{data.entitlement.sugarKg} kg</div>
                                    <p className="text-sm text-gray-600 mt-1">Sugar</p>
                                </div>
                            </div>
                        ) : (
                            <p className="text-gray-500">No entitlement data available for this month</p>
                        )}
                    </CardContent>
                </Card>

                {/* Transaction History */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <History className="h-5 w-5" />
                            Transaction History
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        {data?.transactions && data.transactions.length > 0 ? (
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead>Date</TableHead>
                                        <TableHead>Shop</TableHead>
                                        <TableHead>Rice (kg)</TableHead>
                                        <TableHead>Wheat (kg)</TableHead>
                                        <TableHead>Sugar (kg)</TableHead>
                                        <TableHead>Status</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {data.transactions.map((txn, idx) => (
                                        <TableRow key={idx}>
                                            <TableCell>{new Date(txn.dateTime).toLocaleDateString()}</TableCell>
                                            <TableCell>{txn.fps?.name || 'N/A'}</TableCell>
                                            <TableCell>{txn.riceKg}</TableCell>
                                            <TableCell>{txn.wheatKg}</TableCell>
                                            <TableCell>{txn.sugarKg}</TableCell>
                                            <TableCell>
                                                <Badge variant={txn.status === 'CONFIRMED' ? 'default' : 'destructive'}>
                                                    {txn.status}
                                                </Badge>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        ) : (
                            <div className="text-center py-12 text-gray-500">
                                <History className="h-12 w-12 mx-auto mb-4 opacity-50" />
                                <p>No transaction history found</p>
                            </div>
                        )}
                    </CardContent>
                </Card>

                {/* Help Card */}
                <Card className="mt-6 bg-purple-50 border-purple-200">
                    <CardContent className="pt-6">
                        <div className="flex items-start gap-3">
                            <AlertCircle className="h-5 w-5 text-purple-600 mt-0.5" />
                            <div>
                                <p className="font-semibold text-purple-900 mb-1">Need Help?</p>
                                <p className="text-sm text-purple-800">
                                    If you notice any discrepancies in your transaction history or have not received your
                                    entitled rations, please contact your local Fair Price Shop or file a dispute through
                                    the district office.
                                </p>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
