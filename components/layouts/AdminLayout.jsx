'use client';

import { signOut, useSession } from 'next-auth/react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Shield, LayoutDashboard, Store, Activity, AlertTriangle, FileText, LogOut } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function AdminLayout({ children }) {
    const { data: session } = useSession();
    const pathname = usePathname();

    const navItems = [
        { href: '/admin/dashboard', label: 'Dashboard', icon: LayoutDashboard },
        { href: '/admin/shops', label: 'Shops', icon: Store },
        { href: '/admin/transactions', label: 'Transactions', icon: Activity },
        { href: '/admin/alerts', label: 'Alerts', icon: AlertTriangle },
        { href: '/admin/cases', label: 'Cases', icon: FileText },
    ];

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Sidebar */}
            <aside className="fixed left-0 top-0 h-screen w-64 bg-white border-r shadow-sm">
                <div className="p-6">
                    <div className="flex items-center gap-2 mb-8">
                        <Shield className="h-8 w-8 text-blue-600" />
                        <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                            GrainSecure
                        </span>
                    </div>

                    <nav className="space-y-2">
                        {navItems.map((item) => {
                            const isActive = pathname === item.href;
                            return (
                                <Link key={item.href} href={item.href}>
                                    <div
                                        className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${isActive
                                                ? 'bg-blue-50 text-blue-600'
                                                : 'text-gray-700 hover:bg-gray-100'
                                            }`}
                                    >
                                        <item.icon className="h-5 w-5" />
                                        <span className="font-medium">{item.label}</span>
                                    </div>
                                </Link>
                            );
                        })}
                    </nav>
                </div>

                <div className="absolute bottom-0 w-64 p-6 border-t">
                    <div className="mb-4">
                        <p className="text-sm font-medium text-gray-700">{session?.user?.name || 'Admin'}</p>
                        <p className="text-xs text-gray-500">{session?.user?.email}</p>
                        <p className="text-xs text-blue-600 font-semibold mt-1">{session?.user?.role}</p>
                    </div>
                    <Button
                        variant="outline"
                        className="w-full flex items-center gap-2"
                        onClick={() => signOut({ callbackUrl: '/' })}
                    >
                        <LogOut className="h-4 w-4" />
                        Logout
                    </Button>
                </div>
            </aside>

            {/* Main content */}
            <main className="ml-64 p-8">{children}</main>
        </div>
    );
}
