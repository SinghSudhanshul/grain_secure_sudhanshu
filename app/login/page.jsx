'use client';

import { useState } from 'react';
import { signIn } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Shield, Lock, Mail } from 'lucide-react';

export default function LoginPage() {
    const router = useRouter();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            const result = await signIn('credentials', {
                redirect: false,
                email,
                password,
            });

            if (result?.error) {
                setError('Invalid credentials');
                setLoading(false);
                return;
            }

            // Redirect based on role (we'll fetch this from session)
            const response = await fetch('/api/auth/session');
            const session = await response.json();

            if (session?.user?.role) {
                const roleRoutes = {
                    ADMIN: '/admin/dashboard',
                    INSPECTOR: '/inspector/dashboard',
                    DEALER: '/dealer/dashboard',
                    AUDITOR: '/auditor/dashboard',
                    BENEFICIARY: '/beneficiary/portal',
                };

                router.push(roleRoutes[session.user.role] || '/');
            }
        } catch (err) {
            setError('An error occurred');
            setLoading(false);
        }
    };

    const quickLogin = (demoEmail, demoPassword) => {
        setEmail(demoEmail);
        setPassword(demoPassword);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 flex items-center justify-center p-4">
            <div className="w-full max-w-5xl grid md:grid-cols-2 gap-8">
                {/* Left side - Branding */}
                <div className="flex flex-col justify-center text-center md:text-left">
                    <div className="flex items-center gap-3 mb-6 justify-center md:justify-start">
                        <Shield className="h-12 w-12 text-blue-600" />
                        <span className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                            GrainSecure
                        </span>
                    </div>
                    <h1 className="text-3xl font-bold mb-4 text-gray-800">
                        AI-Enabled PDS Monitoring
                    </h1>
                    <p className="text-gray-600 mb-8">
                        Intelligent fraud detection, real-time monitoring, and tamper-proof audit trails
                        for Public Distribution System.
                    </p>
                    <div className="space-y-3 text-sm text-gray-700">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                            <span>Real-time digital twin simulation</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-purple-600 rounded-full"></div>
                            <span>Explainable AI anomaly detection</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-pink-600 rounded-full"></div>
                            <span>Cryptographic audit chain</span>
                        </div>
                    </div>
                </div>

                {/* Right side - Login form */}
                <Card className="shadow-2xl">
                    <CardHeader>
                        <CardTitle className="text-2xl">Login</CardTitle>
                        <CardDescription>
                            Enter your credentials to access the platform
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div className="space-y-2">
                                <label className="text-sm font-medium flex items-center gap-2">
                                    <Mail className="h-4 w-4" />
                                    Email
                                </label>
                                <Input
                                    type="email"
                                    placeholder="your.email@grainsecure.in"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    required
                                />
                            </div>

                            <div className="space-y-2">
                                <label className="text-sm font-medium flex items-center gap-2">
                                    <Lock className="h-4 w-4" />
                                    Password
                                </label>
                                <Input
                                    type="password"
                                    placeholder="Enter password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                />
                            </div>

                            {error && (
                                <div className="text-sm text-red-600 bg-red-50 p-3 rounded-md">
                                    {error}
                                </div>
                            )}

                            <Button type="submit" className="w-full" disabled={loading}>
                                {loading ? 'Logging in...' : 'Login'}
                            </Button>
                        </form>

                        <div className="mt-6">
                            <p className="text-sm text-gray-600 mb-3">Demo Credentials:</p>
                            <div className="grid grid-cols-2 gap-2 text-xs">
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => quickLogin('admin@grainsecure.in', 'admin123')}
                                >
                                    Admin
                                </Button>
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => quickLogin('inspector@grainsecure.in', 'inspector123')}
                                >
                                    Inspector
                                </Button>
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => quickLogin('dealer@grainsecure.in', 'dealer123')}
                                >
                                    Dealer
                                </Button>
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => quickLogin('auditor@grainsecure.in', 'auditor123')}
                                >
                                    Auditor
                                </Button>
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => quickLogin('beneficiary@grainsecure.in', 'beneficiary123')}
                                >
                                    Beneficiary
                                </Button>
                            </div>
                        </div>

                        <div className="mt-6 text-center">
                            <Link href="/" className="text-sm text-blue-600 hover:underline">
                                ← Back to Home
                            </Link>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
