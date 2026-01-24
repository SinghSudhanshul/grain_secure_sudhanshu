'use client';
export const dynamic = 'force-dynamic';


import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { signOut } from 'next-auth/react';
import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Shield, FileText, AlertTriangle, LogOut } from 'lucide-react';

export default function InspectorDashboard() {
    const { data: session, status } = useSession();
    const router = useRouter();
    const [cases, setCases] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (status === 'unauthenticated') {
            router.push('/login');
        } else if (session?.user?.role !== 'INSPECTOR') {
            router.push('/');
        } else {
            fetchCases();
        }
    }, [session, status]);

    const fetchCases = async () => {
        try {
            const res = await fetch('/api/inspector/cases');
            const data = await res.json();
            setCases(data.cases || []);
            setLoading(false);
        } catch (error) {
            console.error('Failed to fetch cases:', error);
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
                            <p className="text-xs text-gray-600">{session?.user?.role}</p>
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
                    <h1 className="text-3xl font-bold mb-2">Inspector Dashboard</h1>
                    <p className="text-gray-600">Case management and fraud investigation</p>
                </div>

                {/* Stats */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-sm text-gray-600">Total Cases</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-3xl font-bold">{cases.length}</div>
                        </CardContent>
                    </Card>
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-sm text-gray-600">Assigned to Me</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-3xl font-bold text-blue-600">
                                {cases.filter(c => c.assignedToId === session?.user?.id).length}
                            </div>
                        </CardContent>
                    </Card>
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-sm text-gray-600">Pending Action</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-3xl font-bold text-orange-600">
                                {cases.filter(c => c.status === 'OPEN' || c.status === 'ASSIGNED').length}
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* Cases List */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <FileText className="h-5 w-5" />
                            Cases
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        {cases.length === 0 ? (
                            <div className="text-center py-12 text-gray-500">
                                <AlertTriangle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                                <p>No cases found</p>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                {cases.map((caseItem) => (
                                    <div
                                        key={caseItem.id}
                                        className="border rounded-lg p-4 hover:shadow-md transition-shadow"
                                    >
                                        <div className="flex justify-between items-start mb-2">
                                            <div>
                                                <h3 className="font-semibold text-lg">{caseItem.alert?.title || 'Case'}</h3>
                                                <p className="text-sm text-gray-600">{caseItem.alert?.description}</p>
                                            </div>
                                            <Badge
                                                variant={
                                                    caseItem.alert?.severity === 'CRITICAL'
                                                        ? 'destructive'
                                                        : 'secondary'
                                                }
                                            >
                                                {caseItem.alert?.severity}
                                            </Badge>
                                        </div>
                                        <div className="flex gap-4 text-sm text-gray-600 mb-3">
                                            <span>Status: <strong>{caseItem.status}</strong></span>
                                            <span>Created: {new Date(caseItem.createdAt).toLocaleDateString()}</span>
                                        </div>
                                        <div className="flex gap-2">
                                            <Link href={`/inspector/cases/${caseItem.id}`}>
                                                <Button size="sm" variant="outline">View Details</Button>
                                            </Link>
                                            {!caseItem.assignedToId && (
                                                <Button size="sm" variant="default">
                                                    Assign to Me
                                                </Button>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
