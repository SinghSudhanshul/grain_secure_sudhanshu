'use client';

import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { signOut } from 'next-auth/react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Shield, CheckCircle, FileText, Download, LogOut, AlertTriangle } from 'lucide-react';

export default function AuditorDashboard() {
    const { data: session, status } = useSession();
    const router = useRouter();
    const [auditLogs, setAuditLogs] = useState([]);
    const [integrity, setIntegrity] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (status === 'unauthenticated') {
            router.push('/login');
        } else if (session?.user?.role !== 'AUDITOR') {
            router.push('/');
        } else {
            fetchAuditData();
        }
    }, [session, status]);

    const fetchAuditData = async () => {
        try {
            const res = await fetch('/api/auditor/audit-logs');
            const data = await res.json();
            setAuditLogs(data.logs || []);
            setIntegrity(data.integrity);
            setLoading(false);
        } catch (error) {
            console.error('Failed to fetch audit data:', error);
            setLoading(false);
        }
    };

    const verifyIntegrity = async () => {
        setLoading(true);
        await fetchAuditData();
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
                            <p className="text-xs text-gray-600">System Auditor</p>
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
                    <h1 className="text-3xl font-bold mb-2">Auditor Dashboard</h1>
                    <p className="text-gray-600">Audit logs and system integrity verification</p>
                </div>

                {/* Integrity Status */}
                <Card className="mb-6 border-2 border-green-200 bg-gradient-to-r from-green-50 to-emerald-50">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <CheckCircle className="h-6 w-6 text-green-600" />
                            Blockchain-Style Audit Chain Integrity
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-2xl font-bold text-green-600">
                                        {integrity?.verified ? '✅ VERIFIED' : '❌ INTEGRITY BREACH DETECTED'}
                                    </p>
                                    <p className="text-sm text-gray-600 mt-1">
                                        Cryptographic hash chain verification (SHA-256)
                                    </p>
                                </div>
                                <Button onClick={verifyIntegrity} variant="outline">
                                    🔍 Re-verify Chain
                                </Button>
                            </div>

                            {integrity && (
                                <div className="grid md:grid-cols-3 gap-4 mt-4">
                                    <div className="bg-white rounded-lg p-4 border">
                                        <p className="text-sm text-gray-600">Total Audit Records</p>
                                        <p className="text-2xl font-bold">{integrity.totalLogs}</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 border">
                                        <p className="text-sm text-gray-600">Verified Hashes</p>
                                        <p className="text-2xl font-bold text-green-600">{integrity.verifiedCount}</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 border">
                                        <p className="text-sm text-gray-600">Chain Start Block</p>
                                        <p className="text-xs font-mono mt-1 truncate">{integrity.genesisHash}</p>
                                    </div>
                                </div>
                            )}

                            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 mt-4">
                                <p className="text-sm text-purple-900">
                                    <strong>🔐 How it works:</strong> Each audit log contains a hash of the previous record, creating an
                                    immutable chain. Any tampering breaks the chain, making fraud detection instant and irreversible.
                                </p>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Quick Actions */}
                <div className="grid md:grid-cols-2 gap-6 mb-6">
                    <Card className="hover:shadow-lg transition-shadow cursor-pointer">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <FileText className="h-5 w-5" />
                                Generate Audit Report
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <p className="text-gray-600 mb-4">
                                Download comprehensive audit report with all transactions, alerts, and case verdicts
                            </p>
                            <Button className="w-full">
                                <Download className="h-4 w-4 mr-2" />
                                Download PDF Report
                            </Button>
                        </CardContent>
                    </Card>

                    <Card className="hover:shadow-lg transition-shadow cursor-pointer">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <AlertTriangle className="h-5 w-5" />
                                Case Review
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <p className="text-gray-600 mb-4">
                                Review all investigation cases and inspector verdicts for compliance
                            </p>
                            <Button className="w-full" variant="outline">
                                View All Cases
                            </Button>
                        </CardContent>
                    </Card>
                </div>

                {/* Audit Logs Table */}
                <Card>
                    <CardHeader>
                        <CardTitle>Recent Audit Logs</CardTitle>
                    </CardHeader>
                    <CardContent>
                        {auditLogs.length > 0 ? (
                            <div className="overflow-x-auto">
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead>Timestamp</TableHead>
                                            <TableHead>Event Type</TableHead>
                                            <TableHead>Previous Hash</TableHead>
                                            <TableHead>Current Hash</TableHead>
                                            <TableHead>Status</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {auditLogs.slice(0, 20).map((log, idx) => (
                                            <TableRow key={idx}>
                                                <TableCell className="text-xs">
                                                    {new Date(log.createdAt).toLocaleString()}
                                                </TableCell>
                                                <TableCell>
                                                    <Badge variant="outline">{log.eventType}</Badge>
                                                </TableCell>
                                                <TableCell className="font-mono text-xs">
                                                    {log.prevHash.slice(0, 12)}...
                                                </TableCell>
                                                <TableCell className="font-mono text-xs">
                                                    {log.currentHash.slice(0, 12)}...
                                                </TableCell>
                                                <TableCell>
                                                    <Badge className="bg-green-100 text-green-700">
                                                        <CheckCircle className="h-3 w-3 mr-1" />
                                                        Valid
                                                    </Badge>
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </div>
                        ) : (
                            <div className="text-center py-12 text-gray-500">
                                <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                                <p>No audit logs found</p>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
