const { PrismaClient } = require('@prisma/client');
const crypto = require('crypto');

const prisma = new PrismaClient();

function hashPassword(password) {
    return crypto.createHash('sha256').update(password).digest('hex');
}

function generateAuditHash(prevHash, eventType, metaJson, createdAt) {
    const data = `${prevHash}${eventType}${JSON.stringify(metaJson)}${createdAt}`;
    return crypto.createHash('sha256').update(data).digest('hex');
}

async function main() {
    console.log('🌱 Seeding database...');

    // Clear existing data
    await prisma.auditLog.deleteMany();
    await prisma.inspectorAction.deleteMany();
    await prisma.case.deleteMany();
    await prisma.alert.deleteMany();
    await prisma.transaction.deleteMany();
    await prisma.stockLog.deleteMany();
    await prisma.entitlement.deleteMany();
    await prisma.beneficiary.deleteMany();
    await prisma.fPSShop.deleteMany();
    await prisma.district.deleteMany();
    await prisma.user.deleteMany();

    console.log('✅ Cleared existing data');

    // Create district
    const district = await prisma.district.create({
        data: {
            name: 'Bangalore Urban',
        },
    });

    console.log('✅ Created district');

    // Create 20 FPS shops
    const shopNames = [
        'MG Road FPS', 'Whitefield FPS', 'Koramangala FPS', 'Indiranagar FPS',
        'Jayanagar FPS', 'Malleswaram FPS', 'Rajaji Nagar FPS', 'Yelahanka FPS',
        'HSR Layout FPS', 'BTM Layout FPS', 'Electronic City FPS', 'Marathahalli FPS',
        'Bannerghatta FPS', 'Hebbal FPS', 'RT Nagar FPS', 'JP Nagar FPS',
        'Basavanagudi FPS', 'Vijayanagar FPS', 'Peenya FPS', 'Yeshwanthpur FPS',
    ];

    const zones = ['North', 'South', 'East', 'West', 'Central'];
    const shops = [];

    for (let i = 0; i < 20; i++) {
        const shop = await prisma.fPSShop.create({
            data: {
                shopCode: `FPS${String(i + 1).padStart(3, '0')}`,
                name: shopNames[i],
                zone: zones[i % zones.length],
                address: `${shopNames[i]} Area, Bangalore`,
                lat: 12.9716 + (Math.random() - 0.5) * 0.2,
                lng: 77.5946 + (Math.random() - 0.5) * 0.2,
                districtId: district.id,
                riskScore: Math.random() * 100,
            },
        });
        shops.push(shop);
    }

    console.log('✅ Created 20 FPS shops');

    // Create 400 beneficiaries
    const firstNames = ['Rajesh', 'Priya', 'Amit', 'Sunita', 'Ravi', 'Lakshmi', 'Kumar', 'Savita', 'Suresh', 'Anita'];
    const lastNames = ['Kumar', 'Singh', 'Reddy', 'Sharma', 'Patel', 'Nair', 'Rao', 'Gupta', 'Iyer', 'Das'];
    const beneficiaries = [];

    for (let i = 0; i < 400; i++) {
        const firstName = firstNames[Math.floor(Math.random() * firstNames.length)];
        const lastName = lastNames[Math.floor(Math.random() * lastNames.length)];
        const name = `${firstName} ${lastName}`;

        const beneficiary = await prisma.beneficiary.create({
            data: {
                rationCardId: `RC${String(i + 1).padStart(6, '0')}`,
                name,
                age: 20 + Math.floor(Math.random() * 60),
                gender: Math.random() > 0.5 ? 'M' : 'F',
                familySize: 2 + Math.floor(Math.random() * 6),
                aadhaarMasked: `XXXX-XXXX-${Math.floor(1000 + Math.random() * 9000)}`,
                address: `Street ${i + 1}, ${shops[i % shops.length].zone} Zone, Bangalore`,
                phoneNumber: `98${Math.floor(10000000 + Math.random() * 90000000)}`,
                refPhotoUrl: `/photos/beneficiary_${i + 1}.jpg`,
                districtId: district.id,
            },
        });
        beneficiaries.push(beneficiary);
    }

    console.log('✅ Created 400 beneficiaries');

    // Create users for each role
    const users = await Promise.all([
        prisma.user.create({
            data: {
                email: 'admin@grainsecure.in',
                passwordHash: hashPassword('admin123'),
                role: 'ADMIN',
                name: 'District Officer',
            },
        }),
        prisma.user.create({
            data: {
                email: 'inspector@grainsecure.in',
                passwordHash: hashPassword('inspector123'),
                role: 'INSPECTOR',
                name: 'Food Inspector',
            },
        }),
        prisma.user.create({
            data: {
                email: 'dealer@grainsecure.in',
                passwordHash: hashPassword('dealer123'),
                role: 'DEALER',
                name: 'Shop Dealer',
            },
        }),
        prisma.user.create({
            data: {
                email: 'auditor@grainsecure.in',
                passwordHash: hashPassword('auditor123'),
                role: 'AUDITOR',
                name: 'System Auditor',
            },
        }),
        prisma.user.create({
            data: {
                email: 'beneficiary@grainsecure.in',
                passwordHash: hashPassword('beneficiary123'),
                role: 'BENEFICIARY',
                name: 'Sample Beneficiary',
                beneficiaryId: beneficiaries[0].id,
            },
        }),
    ]);

    console.log('✅ Created 5 users with demo credentials');

    // Create entitlements for 6 months
    const months = [];
    for (let i = 0; i < 6; i++) {
        const date = new Date();
        date.setMonth(date.getMonth() - i);
        months.push(date.toISOString().slice(0, 7));
    }

    let entitlementCount = 0;
    for (const beneficiary of beneficiaries) {
        for (const month of months) {
            await prisma.entitlement.create({
                data: {
                    beneficiaryId: beneficiary.id,
                    month,
                    riceKg: 5 * beneficiary.familySize,
                    wheatKg: 4 * beneficiary.familySize,
                    sugarKg: 1 * beneficiary.familySize,
                },
            });
            entitlementCount++;
        }
    }

    console.log(`✅ Created ${entitlementCount} entitlements`);

    // Create stock logs
    for (const shop of shops) {
        // Initial stock delivery
        await prisma.stockLog.create({
            data: {
                fpsId: shop.id,
                riceIn: 5000 + Math.floor(Math.random() * 2000),
                wheatIn: 4000 + Math.floor(Math.random() * 1500),
                sugarIn: 800 + Math.floor(Math.random() * 300),
                dateTime: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
            },
        });

        // Subsequent deliveries
        for (let i = 0; i < 5; i++) {
            await prisma.stockLog.create({
                data: {
                    fpsId: shop.id,
                    riceIn: 1000 + Math.floor(Math.random() * 500),
                    wheatIn: 800 + Math.floor(Math.random() * 400),
                    sugarIn: 200 + Math.floor(Math.random() * 100),
                    dateTime: new Date(Date.now() - (25 - i * 5) * 24 * 60 * 60 * 1000),
                },
            });
        }
    }

    console.log('✅ Created stock logs');

    // Create 10,000 transactions (5% fraudulent)
    console.log('Creating 10,000 transactions...');

    for (let i = 0; i < 10000; i++) {
        const beneficiary = beneficiaries[Math.floor(Math.random() * beneficiaries.length)];
        const shop = shops[Math.floor(Math.random() * shops.length)];
        const daysAgo = Math.floor(Math.random() * 30);
        const dateTime = new Date(Date.now() - daysAgo * 24 * 60 * 60 * 1000);

        const currentMonth = dateTime.toISOString().slice(0, 7);
        const entitlement = await prisma.entitlement.findFirst({
            where: { beneficiaryId: beneficiary.id, month: currentMonth },
        });

        if (!entitlement) continue;

        // Determine if this is a fraudulent transaction (5%)
        const isFraud = Math.random() < 0.05;

        let riceKg, wheatKg, sugarKg, authMethod, authStatus, riskScore, anomalyType, anomalyFlags;

        if (isFraud) {
            // Over-withdrawal
            riceKg = entitlement.riceKg * (1.2 + Math.random() * 0.5);
            wheatKg = entitlement.wheatKg * (1.2 + Math.random() * 0.5);
            sugarKg = entitlement.sugarKg * (1.2 + Math.random() * 0.5);
            authMethod = Math.random() > 0.5 ? 'MANUAL' : 'OTP';
            authStatus = Math.random() > 0.7 ? 'FAILED' : 'SUCCESS';
            riskScore = 50 + Math.floor(Math.random() * 50);
            anomalyType = 'OVER_WITHDRAWAL, HIGH_FREQUENCY';
            anomalyFlags = JSON.stringify(['Over-withdrawal detected', 'Suspicious pattern']);
        } else {
            // Normal transaction
            const factor = 0.7 + Math.random() * 0.3;
            riceKg = parseFloat((entitlement.riceKg * factor).toFixed(2));
            wheatKg = parseFloat((entitlement.wheatKg * factor).toFixed(2));
            sugarKg = parseFloat((entitlement.sugarKg * factor).toFixed(2));
            authMethod = ['OTP', 'FACE'][Math.floor(Math.random() * 2)];
            authStatus = 'SUCCESS';
            riskScore = Math.floor(Math.random() * 30);
            anomalyType = 'NONE';
            anomalyFlags = JSON.stringify([]);
        }

        await prisma.transaction.create({
            data: {
                beneficiaryId: beneficiary.id,
                fpsId: shop.id,
                dateTime,
                riceKg,
                wheatKg,
                sugarKg,
                authMethod,
                authStatus,
                riskScore,
                anomalyType,
                anomalyFlags,
            },
        });

        if (i % 1000 === 0) {
            console.log(`  Created ${i} transactions...`);
        }
    }

    console.log('✅ Created 10,000 transactions (5% fraudulent)');

    // Create alerts for high-risk transactions
    const highRiskTransactions = await prisma.transaction.findMany({
        where: { riskScore: { gte: 50 } },
        take: 50,
        include: { beneficiary: true, fps: true },
    });

    for (const txn of highRiskTransactions) {
        const severity = txn.riskScore >= 80 ? 'CRITICAL' : txn.riskScore >= 65 ? 'HIGH' : 'MEDIUM';

        await prisma.alert.create({
            data: {
                transactionId: txn.id,
                beneficiaryId: txn.beneficiaryId,
                fpsId: txn.fpsId,
                severity,
                title: `${severity} Risk Transaction`,
                description: `Suspicious transaction at ${txn.fps.name} for ${txn.beneficiary.name}`,
                evidence: txn.anomalyFlags,
            },
        });
    }

    console.log('✅ Created alerts');

    // Create cases for critical alerts
    const criticalAlerts = await prisma.alert.findMany({
        where: { severity: 'CRITICAL' },
        take: 10,
    });

    const inspector = users.find(u => u.role === 'INSPECTOR');

    for (const alert of criticalAlerts) {
        const caseRecord = await prisma.case.create({
            data: {
                alertId: alert.id,
                status: ['OPEN', 'ASSIGNED', 'INVESTIGATING'][Math.floor(Math.random() * 3)],
                assignedToId: Math.random() > 0.5 ? inspector.id : null,
                verdict: null,
                notes: 'Under investigation',
            },
        });

        // Add inspector action
        if (caseRecord.assignedToId) {
            await prisma.inspectorAction.create({
                data: {
                    caseId: caseRecord.id,
                    inspectorId: inspector.id,
                    actionType: 'ASSIGNED',
                    notes: 'Case assigned for investigation',
                },
            });
        }
    }

    console.log('✅ Created cases');

    // Create audit logs with hash chain
    let prevHash = '0000000000000000000000000000000000000000000000000000000000000000';

    for (let i = 0; i < 100; i++) {
        const eventTypes = ['USER_LOGIN', 'TRANSACTION_CREATED', 'ALERT_GENERATED', 'CASE_STATUS_CHANGED'];
        const eventType = eventTypes[Math.floor(Math.random() * eventTypes.length)];
        const createdAt = new Date(Date.now() - (100 - i) * 60 * 60 * 1000);
        const metaJson = JSON.stringify({
            eventId: i + 1,
            userId: users[Math.floor(Math.random() * users.length)].id,
            timestamp: createdAt.toISOString(),
        });

        const currentHash = generateAuditHash(prevHash, eventType, metaJson, createdAt.toISOString());

        await prisma.auditLog.create({
            data: {
                eventType,
                metaJson,
                prevHash,
                currentHash,
                createdAt,
            },
        });

        prevHash = currentHash;
    }

    console.log('✅ Created tamper-proof audit logs with hash chain');

    console.log('\n🎉 Database seeded successfully!');
    console.log('\n📊 Summary:');
    console.log(`  Districts: 1`);
    console.log(`  FPS Shops: 20`);
    console.log(`  Beneficiaries: 400`);
    console.log(`  Users: 5`);
    console.log(`  Entitlements: ${entitlementCount}`);
    console.log(`  Transactions: 10,000`);
    console.log(`  Alerts: ${highRiskTransactions.length}`);
    console.log(`  Cases: ${criticalAlerts.length}`);
    console.log(`  Audit Logs: 100`);
    console.log('\n🔑 Demo Credentials:');
    console.log('  admin@grainsecure.in / admin123');
    console.log('  inspector@grainsecure.in / inspector123');
    console.log('  dealer@grainsecure.in / dealer123');
    console.log('  auditor@grainsecure.in / auditor123');
    console.log('  beneficiary@grainsecure.in / beneficiary123');
}

main()
    .catch((e) => {
        console.error('❌ Seeding failed:', e);
        process.exit(1);
    })
    .finally(async () => {
        await prisma.$disconnect();
    });
