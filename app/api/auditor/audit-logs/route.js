import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-config';
import { prisma } from '@/lib/prisma';
import crypto from 'crypto';

function verifyHash(prevHash, eventType, metaJson, createdAt, currentHash) {
    const data = `${prevHash}${eventType}${JSON.stringify(metaJson)}${createdAt}`;
    const computed = crypto.createHash('sha256').update(data).digest('hex');
    return computed === currentHash;
}

export async function GET() {
    try {
        const session = await getServerSession(authOptions);

        if (!session || session.user.role !== 'AUDITOR') {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        // Fetch all audit logs
        const logs = await prisma.auditLog.findMany({
            orderBy: { createdAt: 'asc' },
        });

        // Verify hash chain integrity
        let verified = true;
        let verifiedCount = 0;

        for (let i = 0; i < logs.length; i++) {
            const log = logs[i];
            const meta = typeof log.metaJson === 'string' ? JSON.parse(log.metaJson) : log.metaJson;

            const isValid = verifyHash(
                log.prevHash,
                log.eventType,
                meta,
                log.createdAt.toISOString(),
                log.currentHash
            );

            if (isValid) {
                verifiedCount++;
            } else {
                verified = false;
                console.warn(`Hash chain broken at log ${log.id}`);
            }
        }

        const genesisHash = logs.length > 0 ? logs[0].prevHash : '0'.repeat(64);

        return NextResponse.json({
            logs,
            integrity: {
                verified,
                totalLogs: logs.length,
                verifiedCount,
                genesisHash,
            },
        });
    } catch (error) {
        console.error('Audit logs API error:', error);
        return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
    }
}
