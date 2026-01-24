import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';
import { analyzeTransaction } from '@/lib/fraud-detection';

// Socket.IO broadcasts removed - implement with Server-Sent Events or separate WebSocket server
const broadcastTransaction = (data) => console.log('Transaction:', data);
const broadcastAlert = (data) => console.log('Alert:', data);
const broadcastSimulatorStatus = (data) => console.log('Simulator status:', data);


let simulatorInterval = null;
let isRunning = false;

export async function POST(request) {
    try {
        const { action } = await request.json();

        if (action === 'start') {
            if (isRunning) {
                return NextResponse.json({ message: 'Already running' });
            }

            isRunning = true;
            broadcastSimulatorStatus({ running: true });

            // Start generating transactions every 2-5 seconds
            simulatorInterval = setInterval(async () => {
                await generateTransaction();
            }, Math.random() * 3000 + 2000);

            return NextResponse.json({ message: 'Simulator started' });
        }

        if (action === 'stop') {
            if (simulatorInterval) {
                clearInterval(simulatorInterval);
                simulatorInterval = null;
            }
            isRunning = false;
            broadcastSimulatorStatus({ running: false });

            return NextResponse.json({ message: 'Simulator stopped' });
        }

        return NextResponse.json({ error: 'Invalid action' }, { status: 400 });
    } catch (error) {
        console.error('Simulator control error:', error);
        return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
    }
}

async function generateTransaction() {
    try {
        // Get random beneficiary and shop
        const beneficiaries = await prisma.beneficiary.findMany({ take: 50 });
        const shops = await prisma.fPSShop.findMany({ take: 20 });

        if (beneficiaries.length === 0 || shops.length === 0) {
            return;
        }

        const beneficiary = beneficiaries[Math.floor(Math.random() * beneficiaries.length)];
        const shop = shops[Math.floor(Math.random() * shops.length)];

        // Get entitlement for current month
        const currentMonth = new Date().toISOString().slice(0, 7);
        const entitlement = await prisma.entitlement.findFirst({
            where: { beneficiaryId: beneficiary.id, month: currentMonth },
        });

        if (!entitlement) {
            return;
        }

        // Generate realistic quantities (70-100% of entitlement normally)
        const factor = 0.7 + Math.random() * 0.3;
        const riceKg = parseFloat((entitlement.riceKg * factor).toFixed(2));
        const wheatKg = parseFloat((entitlement.wheatKg * factor).toFixed(2));
        const sugarKg = parseFloat((entitlement.sugarKg * factor).toFixed(2));

        const authMethods = ['OTP', 'FACE', 'MANUAL'];
        const authMethod = authMethods[Math.floor(Math.random() * authMethods.length)];
        const authStatus = Math.random() > 0.95 ? 'FAILED' : 'SUCCESS';

        // Get recent transactions for analysis
        const recentTransactions = await prisma.transaction.findMany({
            where: { beneficiaryId: beneficiary.id },
            orderBy: { dateTime: 'desc' },
            take: 10,
        });

        // Create transaction
        const transaction = await prisma.transaction.create({
            data: {
                beneficiaryId: beneficiary.id,
                fpsId: shop.id,
                riceKg,
                wheatKg,
                sugarKg,
                authMethod,
                authStatus,
                dateTime: new Date(),
            },
        });

        // Run AI analysis
        const analysis = await analyzeTransaction(transaction, {
            beneficiary,
            entitlement,
            recentTransactions,
            shopBaseline: {
                avgQuantity: 15,
                quantityStdDev: 3,
                dailyAvg: 50,
            },
            allBeneficiaries: [],
        });

        // Update transaction with analysis results
        const updatedTransaction = await prisma.transaction.update({
            where: { id: transaction.id },
            data: {
                riskScore: analysis.riskScore,
                anomalyType: analysis.anomalyType,
                anomalyFlags: JSON.stringify(analysis.evidence),
            },
        });

        // Broadcast transaction
        broadcastTransaction({
            beneficiaryName: beneficiary.name,
            shopName: shop.name,
            dateTime: transaction.dateTime,
            riceKg,
            wheatKg,
            sugarKg,
        });

        // Create alert if high risk
        if (analysis.riskScore >= 50) {
            const alert = await prisma.alert.create({
                data: {
                    transactionId: transaction.id,
                    beneficiaryId: beneficiary.id,
                    fpsId: shop.id,
                    severity: analysis.severity,
                    title: `${analysis.severity} Risk: ${analysis.anomalyType}`,
                    description: analysis.evidence.join('; '),
                    evidence: JSON.stringify(analysis.evidence),
                },
            });

            broadcastAlert({
                title: alert.title,
                description: alert.description,
                severity: alert.severity,
                createdAt: alert.createdAt,
            });

            // Auto-create case for critical alerts
            if (analysis.severity === 'CRITICAL') {
                await prisma.case.create({
                    data: {
                        alertId: alert.id,
                        status: 'OPEN',
                    },
                });
            }
        }
    } catch (error) {
        console.error('Transaction generation error:', error);
    }
}
