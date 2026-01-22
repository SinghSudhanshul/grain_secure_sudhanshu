import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/app/api/auth/[...nextauth]/route';
import { prisma } from '@/lib/prisma';

export async function GET() {
    try {
        const session = await getServerSession(authOptions);

        if (!session || session.user.role !== 'INSPECTOR') {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const cases = await prisma.case.findMany({
            include: {
                alert: true,
                assignedTo: { select: { name: true, email: true } },
            },
            orderBy: { createdAt: 'desc' },
        });

        return NextResponse.json({ cases });
    } catch (error) {
        console.error('Inspector cases API error:', error);
        return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
    }
}
