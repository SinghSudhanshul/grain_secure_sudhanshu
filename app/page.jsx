import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Shield, Brain, Activity, FileCheck, Users, AlertTriangle } from 'lucide-react';

export default function HomePage() {
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
                    <nav className="flex gap-4">
                        <Link href="/login">
                            <Button variant="outline">Login</Button>
                        </Link>
                        <Link href="/public/dashboard">
                            <Button>Public Dashboard</Button>
                        </Link>
                    </nav>
                </div>
            </header>

            {/* Hero */}
            <section className="container mx-auto px-4 py-20 text-center">
                <div className="max-w-4xl mx-auto">
                    <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
                        AI-Enabled Intelligent PDS Monitoring Platform
                    </h1>
                    <p className="text-xl text-gray-600 mb-8">
                        Eliminating leakages, ghost beneficiaries, and corruption in Public Distribution System
                        with real-time AI fraud detection and tamper-proof audit trails.
                    </p>
                    <div className="flex gap-4 justify-center">
                        <Link href="/login">
                            <Button size="lg" className="text-lg px-8">
                                Get Started
                            </Button>
                        </Link>
                        <Link href="#demo">
                            <Button size="lg" variant="outline" className="text-lg px-8">
                                Live Demo
                            </Button>
                        </Link>
                    </div>
                </div>
            </section>

            {/* Problem Statement */}
            <section className="bg-red-50 py-16">
                <div className="container mx-auto px-4">
                    <div className="max-w-4xl mx-auto">
                        <h2 className="text-3xl font-bold text-center mb-8 flex items-center justify-center gap-2">
                            <AlertTriangle className="h-8 w-8 text-red-600" />
                            The Crisis
                        </h2>
                        <div className="grid md:grid-cols-3 gap-6">
                            <Card className="border-red-200">
                                <CardHeader>
                                    <CardTitle className="text-red-700">₹50,000 Crore Loss</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    Annual leakage in India's PDS due to stock diversion and ghost beneficiaries
                                </CardContent>
                            </Card>
                            <Card className="border-red-200">
                                <CardHeader>
                                    <CardTitle className="text-red-700">23% Diversion Rate</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    Nearly 1 in 4 grains allocated never reach genuine beneficiaries
                                </CardContent>
                            </Card>
                            <Card className="border-red-200">
                                <CardHeader>
                                    <CardTitle className="text-red-700">Manual Monitoring</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    Traditional inspection is slow, costly, and easily manipulated
                                </CardContent>
                            </Card>
                        </div>
                    </div>
                </div>
            </section>

            {/* Solution */}
            <section className="py-16">
                <div className="container mx-auto px-4">
                    <h2 className="text-4xl font-bold text-center mb-12">Our Solution</h2>
                    <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
                        <Card className="border-blue-200 hover:shadow-lg transition-shadow">
                            <CardHeader>
                                <Brain className="h-12 w-12 text-blue-600 mb-2" />
                                <CardTitle>AI Fraud Detection</CardTitle>
                            </CardHeader>
                            <CardContent>
                                Explainable anomaly detection using statistical models, z-scores, and pattern recognition to flag fraud with evidence
                            </CardContent>
                        </Card>
                        <Card className="border-purple-200 hover:shadow-lg transition-shadow">
                            <CardHeader>
                                <Activity className="h-12 w-12 text-purple-600 mb-2" />
                                <CardTitle>Real-Time Digital Twin</CardTitle>
                            </CardHeader>
                            <CardContent>
                                Live simulation of entire PDS ecosystem with transaction streams, stock movements, and instant anomaly alerts
                            </CardContent>
                        </Card>
                        <Card className="border-green-200 hover:shadow-lg transition-shadow">
                            <CardHeader>
                                <FileCheck className="h-12 w-12 text-green-600 mb-2" />
                                <CardTitle>Tamper-Proof Audit</CardTitle>
                            </CardHeader>
                            <CardContent>
                                Cryptographic hash chaining ensures every transaction and decision is permanently recorded and verifiable
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </section>

            {/* Architecture */}
            <section className="bg-gray-50 py-16">
                <div className="container mx-auto px-4">
                    <h2 className="text-4xl font-bold text-center mb-12">System Architecture</h2>
                    <div className="max-w-5xl mx-auto">
                        <div className="grid md:grid-cols-2 gap-8">
                            <Card>
                                <CardHeader>
                                    <CardTitle>Frontend & Real-Time</CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-2 text-sm">
                                    <div>✅ Next.js 14 App Router</div>
                                    <div>✅ Socket.IO for live updates</div>
                                    <div>✅ Recharts for analytics</div>
                                    <div>✅ Leaflet for fraud heatmaps</div>
                                    <div>✅ shadcn/ui components</div>
                                </CardContent>
                            </Card>
                            <Card>
                                <CardHeader>
                                    <CardTitle>Backend & AI</CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-2 text-sm">
                                    <div>✅ Prisma ORM + SQLite</div>
                                    <div>✅ NextAuth JWT authentication</div>
                                    <div>✅ AI anomaly detection engine</div>
                                    <div>✅ Hash-chained audit logs</div>
                                    <div>✅ PDF report generation</div>
                                </CardContent>
                            </Card>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features */}
            <section id="demo" className="py-16">
                <div className="container mx-auto px-4">
                    <h2 className="text-4xl font-bold text-center mb-12">Key Features</h2>
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
                        <FeatureCard title="Role-Based Access" description="6 distinct roles: Admin, Inspector, Dealer, Auditor, Beneficiary, Public" />
                        <FeatureCard title="Smart Verification" description="OTP, Face match simulation, and logged manual overrides" />
                        <FeatureCard title="Stock Reconciliation" description="Automatic mismatch detection between allotment and distribution" />
                        <FeatureCard title="Case Management" description="Alert → Investigation → Verdict workflow like real governance" />
                        <FeatureCard title="Live Simulation" description="Inject fraud scenarios and watch AI detect them in real-time" />
                        <FeatureCard title="Public Transparency" description="Aggregated district data visible to everyone without login" />
                    </div>
                </div>
            </section>

            {/* Impact */}
            <section className="bg-gradient-to-r from-blue-600 to-purple-600 text-white py-16">
                <div className="container mx-auto px-4 text-center">
                    <h2 className="text-4xl font-bold mb-8">Expected Impact</h2>
                    <div className="grid md:grid-cols-4 gap-8 max-w-5xl mx-auto">
                        <div>
                            <div className="text-5xl font-bold mb-2">60%</div>
                            <div className="text-blue-100">Reduction in leakage</div>
                        </div>
                        <div>
                            <div className="text-5xl font-bold mb-2">₹30K Cr</div>
                            <div className="text-blue-100">Annual savings potential</div>
                        </div>
                        <div>
                            <div className="text-5xl font-bold mb-2">100%</div>
                            <div className="text-blue-100">Audit trail coverage</div>
                        </div>
                        <div>
                            <div className="text-5xl font-bold mb-2">Real-time</div>
                            <div className="text-blue-100">Fraud detection</div>
                        </div>
                    </div>
                </div>
            </section>

            {/* CTA */}
            <section className="py-20">
                <div className="container mx-auto px-4 text-center">
                    <h2 className="text-4xl font-bold mb-6">Ready to See It in Action?</h2>
                    <p className="text-xl text-gray-600 mb-8">
                        Login with demo credentials or explore the public dashboard
                    </p>
                    <div className="flex gap-4 justify-center">
                        <Link href="/login">
                            <Button size="lg" className="text-lg px-8">
                                Login as Admin
                            </Button>
                        </Link>
                        <Link href="/public/dashboard">
                            <Button size="lg" variant="outline" className="text-lg px-8">
                                View Public Dashboard
                            </Button>
                        </Link>
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="border-t bg-gray-50 py-8">
                <div className="container mx-auto px-4 text-center text-gray-600">
                    <p className="mb-2 flex items-center justify-center gap-2">
                        <Shield className="h-5 w-5" />
                        GrainSecure - Smart India Hackathon 2026
                    </p>
                    <p className="text-sm">
                        Built with Next.js, Prisma, Socket.IO, and AI • Production-ready PDS monitoring platform
                    </p>
                </div>
            </footer>
        </div>
    );
}

function FeatureCard({ title, description }) {
    return (
        <Card className="hover:shadow-md transition-shadow">
            <CardHeader>
                <CardTitle className="text-lg">{title}</CardTitle>
            </CardHeader>
            <CardContent>
                <p className="text-sm text-gray-600">{description}</p>
            </CardContent>
        </Card>
    );
}
