import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/app/api/auth/[...nextauth]/route';
import { prisma } from '@/lib/prisma';

export async function GET() {
    try {
        const session = await getServerSession(authOptions);

        if (!session || session.user.role !== 'DEALER') {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        // For demo: use first shop
        const shop = await prisma.fPSShop.findFirst();

        if (!shop) {
            return NextResponse.json({
                stock: { rice: 0, wheat: 0, sugar: 0 },
                todayCount: 0,
                complianceScore: 100,
            });
        }

        // Calculate stock
        const stockLogs = await prisma.stockLog.findMany({
            where: { fpsId: shop.id },
        });

        const stockIn = stockLogs.reduce(
            (acc, log) => ({
                rice: acc.rice + log.riceIn,
                wheat: acc.wheat + log.wheatIn,
                sugar: acc.sugar + log.sugarIn,
            }),
            { rice: 0, wheat: 0, sugar: 0 }
        );

        const transactions = await prisma.transaction.findMany({
            where: { fpsId: shop.id },
        });

        const distributed = transactions.reduce(
            (acc, txn) => ({
                rice: acc.rice + txn.riceKg,
                wheat: acc.wheat + txn.wheatKg,
                sugar: acc.sugar + txn.sugarKg,
            }),
            { rice: 0, wheat: 0, sugar: 0 }
        );

        const stock = {
            rice: Math.max(0, Math.round(stockIn.rice - distributed.rice)),
            wheat: Math.max(0, Math.round(stockIn.wheat - distributed.wheat)),
            sugar: Math.max(0, Math.round(stockIn.sugar - distributed.sugar)),
        };

        // Today's count
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const todayCount = await prisma.transaction.count({
            where: {
                fpsId: shop.id,
                dateTime: { gte: today },
            },
        });

        const complianceScore = Math.round(100 - shop.riskScore);

        return NextResponse.json({
            stock,
            todayCount,
            complianceScore,
        });
    } catch (error) {
        console.error('Dealer dashboard API error:', error);
        return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
    }
}
