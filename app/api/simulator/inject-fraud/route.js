import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';

export async function POST(request) {
    try {
        const { scenario } = await request.json();

        if (scenario === 'stock_diversion') {
            await injectStockDiversion();
        } else if (scenario === 'ghost_beneficiary') {
            await injectGhostBeneficiary();
        } else if (scenario === 'bulk_spike') {
            await injectBulkSpike();
        }

        return NextResponse.json({ message: 'Fraud scenario injected' });
    } catch (error) {
        console.error('Fraud injection error:', error);
        return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
    }
}

async function injectStockDiversion() {
    // Create shop with stock mismatch
    const shops = await prisma.fPSShop.findMany({ take: 5 });
    const shop = shops[Math.floor(Math.random() * shops.length)];

    // Log inflated incoming stock
    await prisma.stockLog.create({
        data: {
            fpsId: shop.id,
            riceIn: 1000,
            wheatIn: 800,
            sugarIn: 200,
            dateTime: new Date(),
        },
    });

    // But only distribute a fraction
    const beneficiaries = await prisma.beneficiary.findMany({ take: 10 });

    for (let i = 0; i < 5; i++) {
        const beneficiary = beneficiaries[i];
        await prisma.transaction.create({
            data: {
                beneficiaryId: beneficiary.id,
                fpsId: shop.id,
                riceKg: 5,
                wheatKg: 3,
                sugarKg: 1,
                authMethod: 'OTP',
                authStatus: 'SUCCESS',
                riskScore: 75,
                anomalyType: 'STOCK_DIVERSION',
                anomalyFlags: JSON.stringify(['Stock mismatch detected: 85% discrepancy']),
                dateTime: new Date(),
            },
        });
    }

    // Create alert
    await prisma.alert.create({
        data: {
            fpsId: shop.id,
            severity: 'CRITICAL',
            title: '🚨 Stock Diversion Detected',
            description: `Shop ${shop.name} shows 85% stock mismatch - possible diversion`,
            evidence: JSON.stringify(['Incoming: 1000kg rice', 'Distributed: only 25kg', 'Mismatch: 975kg']),
        },
    });
}

async function injectGhostBeneficiary() {
    // Create duplicate transactions from similar beneficiary
    const beneficiaries = await prisma.beneficiary.findMany({ take: 10 });
    const beneficiary = beneficiaries[0];
    const shop = await prisma.fPSShop.findFirst();

    // Create multiple suspicious transactions
    for (let i = 0; i < 3; i++) {
        await prisma.transaction.create({
            data: {
                beneficiaryId: beneficiary.id,
                fpsId: shop.id,
                riceKg: 10,
                wheatKg: 8,
                sugarKg: 2,
                authMethod: 'MANUAL',
                authStatus: 'FAILED',
                riskScore: 85,
                anomalyType: 'DUPLICATE_BENEFICIARY, AUTH_FAILURE',
                anomalyFlags: JSON.stringify([
                    'Manual override used',
                    'Authentication failed',
                    'Potential duplicate beneficiary detected',
                ]),
                dateTime: new Date(),
            },
        });
    }

    await prisma.alert.create({
        data: {
            beneficiaryId: beneficiary.id,
            fpsId: shop.id,
            severity: 'CRITICAL',
            title: '👻 Ghost Beneficiary Pattern',
            description: `Multiple failed auth + manual overrides for ${beneficiary.name}`,
            evidence: JSON.stringify([
                '3 transactions in short period',
                'All authentication failed',
                'Manual override suspicious',
            ]),
        },
    });
}

async function injectBulkSpike() {
    const shop = await prisma.fPSShop.findFirst();
    const beneficiaries = await prisma.beneficiary.findMany({ take: 20 });

    // Create sudden bulk distributions
    for (let i = 0; i < 15; i++) {
        const beneficiary = beneficiaries[i];
        await prisma.transaction.create({
            data: {
                beneficiaryId: beneficiary.id,
                fpsId: shop.id,
                riceKg: 25,
                wheatKg: 20,
                sugarKg: 5,
                authMethod: 'OTP',
                authStatus: 'SUCCESS',
                riskScore: 65,
                anomalyType: 'BULK_SPIKE',
                anomalyFlags: JSON.stringify(['Sudden bulk distribution spike: 5x normal rate']),
                dateTime: new Date(),
            },
        });
    }

    await prisma.alert.create({
        data: {
            fpsId: shop.id,
            severity: 'HIGH',
            title: '📈 Bulk Distribution Spike',
            description: `Shop ${shop.name} distributed to 15 beneficiaries in minutes`,
            evidence: JSON.stringify([
                '15 transactions in <5 minutes',
                '5x normal distribution rate',
                'Suspicious timing pattern',
            ]),
        },
    });
}
