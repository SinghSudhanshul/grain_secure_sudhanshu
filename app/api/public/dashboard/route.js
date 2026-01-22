import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';

export async function GET() {
    try {
        // Public aggregated data only - no personal information
        const [
            totalBeneficiaries,
            totalShops,
            totalTransactions,
            totalAnomalies,
        ] = await Promise.all([
            prisma.beneficiary.count(),
            prisma.fPSShop.count(),
            prisma.transaction.count(),
            prisma.transaction.count({ where: { riskScore: { gt: 30 } } }),
        ]);

        // Calculate leakage prevented
        const highRiskTransactions = await prisma.transaction.findMany({
            where: { riskScore: { gt: 50 } },
            select: { riceKg: true, wheatKg: true, sugarKg: true },
        });

        const leakagePrevented = highRiskTransactions.reduce((sum, txn) => {
            return sum + (txn.riceKg * 20) + (txn.wheatKg * 15) + (txn.sugarKg * 35);
        }, 0);

        // Top compliant shops (lowest risk score)
        const topShops = await prisma.fPSShop.findMany({
            orderBy: { riskScore: 'asc' },
            take: 10,
            select: { name: true, zone: true, riskScore: true },
        });

        const topShopsWithCompliance = topShops.map(shop => ({
            name: shop.name,
            zone: shop.zone,
            complianceScore: Math.round(100 - shop.riskScore),
        }));

        return NextResponse.json({
            totalBeneficiaries,
            totalShops,
            totalTransactions,
            totalAnomalies,
            leakagePrevented: Math.round(leakagePrevented),
            topShops: topShopsWithCompliance,
        });
    } catch (error) {
        console.error('Public dashboard API error:', error);
        return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
    }
}
