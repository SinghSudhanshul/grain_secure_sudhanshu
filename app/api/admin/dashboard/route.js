import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/app/api/auth/[...nextauth]/route';
import { prisma } from '@/lib/prisma';

export async function GET() {
    try {
        const session = await getServerSession(authOptions);

        if (!session || session.user.role !== 'ADMIN') {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        // Fetch statistics
        const [
            totalBeneficiaries,
            totalShops,
            totalTransactions,
            totalAnomalies,
            openCases,
            alerts,
            transactions,
        ] = await Promise.all([
            prisma.beneficiary.count(),
            prisma.fPSShop.count(),
            prisma.transaction.count(),
            prisma.transaction.count({ where: { riskScore: { gt: 30 } } }),
            prisma.case.count({ where: { status: { in: ['OPEN', 'ASSIGNED', 'INVESTIGATING'] } } }),
            prisma.alert.findMany({
                take: 10,
                orderBy: { createdAt: 'desc' },
                include: {
                    beneficiary: { select: { name: true } },
                    fps: { select: { name: true } },
                },
            }),
            prisma.transaction.findMany({
                take: 10,
                orderBy: { dateTime: 'desc' },
                include: {
                    beneficiary: { select: { name: true } },
                    fps: { select: { name: true } },
                },
            }),
        ]);

        // Calculate estimated leakage prevented
        const highRiskTransactions = await prisma.transaction.findMany({
            where: { riskScore: { gt: 50 } },
            select: { riceKg: true, wheatKg: true, sugarKg: true },
        });

        const leakagePrevented = highRiskTransactions.reduce((sum, txn) => {
            // Assuming ₹20/kg rice, ₹15/kg wheat, ₹35/kg sugar
            return sum + (txn.riceKg * 20) + (txn.wheatKg * 15) + (txn.sugarKg * 35);
        }, 0);

        // Get anomaly trend (last 7 days)
        const sevenDaysAgo = new Date();
        sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);

        const anomalyTrend = [];
        for (let i = 6; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            const dateStr = date.toISOString().split('T')[0];

            const count = await prisma.transaction.count({
                where: {
                    dateTime: {
                        gte: new Date(dateStr),
                        lt: new Date(new Date(dateStr).getTime() + 24 * 60 * 60 * 1000),
                    },
                    riskScore: { gt: 30 },
                },
            });

            anomalyTrend.push({ date: dateStr.slice(5), count });
        }

        // Risk by shop (top 5)
        const shops = await prisma.fPSShop.findMany({
            orderBy: { riskScore: 'desc' },
            take: 5,
            select: { name: true, riskScore: true },
        });

        const riskByShop = shops.map(s => ({
            shop: s.name.slice(0, 15),
            riskScore: Math.round(s.riskScore),
        }));

        // Anomaly distribution
        const anomalyTypes = await prisma.transaction.groupBy({
            by: ['anomalyType'],
            where: { anomalyType: { not: null } },
            _count: true,
        });

        const anomalyDistribution = anomalyTypes
            .filter(a => a.anomalyType && a.anomalyType !== 'NONE')
            .slice(0, 4)
            .map(a => ({
                name: a.anomalyType.split(',')[0].replace(/_/g, ' '),
                value: a._count,
            }));

        return NextResponse.json({
            stats: {
                totalBeneficiaries,
                totalShops,
                totalTransactions,
                totalAnomalies,
                openCases,
                leakagePrevented: Math.round(leakagePrevented),
                anomalyTrend,
                riskByShop,
                anomalyDistribution: anomalyDistribution.length > 0 ? anomalyDistribution : [
                    { name: 'Over Withdrawal', value: 12 },
                    { name: 'High Frequency', value: 8 },
                    { name: 'Stock Mismatch', value: 5 },
                    { name: 'Auth Failure', value: 3 },
                ],
            },
            recentAlerts: alerts.map(a => ({
                title: a.title,
                description: a.description,
                severity: a.severity,
                createdAt: a.createdAt,
            })),
            recentTransactions: transactions.map(t => ({
                beneficiaryName: t.beneficiary.name,
                shopName: t.fps.name,
                dateTime: t.dateTime,
                riceKg: t.riceKg,
                wheatKg: t.wheatKg,
                sugarKg: t.sugarKg,
            })),
        });
    } catch (error) {
        console.error('Dashboard API error:', error);
        return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
    }
}
