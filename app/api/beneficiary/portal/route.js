import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-config';
import { prisma } from '@/lib/prisma';

export async function GET() {
    try {
        const session = await getServerSession(authOptions);

        if (!session || session.user.role !== 'BENEFICIARY') {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        // Get user's beneficiary record
        const user = await prisma.user.findUnique({
            where: { email: session.user.email },
            include: { beneficiary: true },
        });

        if (!user?.beneficiary) {
            return NextResponse.json({ error: 'Beneficiary not found' }, { status: 404 });
        }

        // Get current month entitlement
        const currentMonth = new Date().toISOString().slice(0, 7);
        const entitlement = await prisma.entitlement.findFirst({
            where: {
                beneficiaryId: user.beneficiary.id,
                month: currentMonth,
            },
        });

        // Get transaction history
        const transactions = await prisma.transaction.findMany({
            where: { beneficiaryId: user.beneficiary.id },
            include: { fps: { select: { name: true } } },
            orderBy: { dateTime: 'desc' },
            take: 20,
        });

        return NextResponse.json({
            beneficiary: user.beneficiary,
            entitlement,
            transactions,
        });
    } catch (error) {
        console.error('Beneficiary portal API error:', error);
        return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
    }
}
