/**
 * GrainSecure AI Fraud Detection Engine
 * Explainable anomaly detection using statistical methods and rule-based patterns
 */

// Statistical helper functions
function mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function stdDev(arr) {
    const avg = mean(arr);
    const squareDiffs = arr.map(value => Math.pow(value - avg, 2));
    return Math.sqrt(mean(squareDiffs));
}

function zScore(value, arr) {
    const avg = mean(arr);
    const sd = stdDev(arr);
    return sd === 0 ? 0 : (value - avg) / sd;
}

/**
 * Analyze transaction for fraud patterns
 */
export async function analyzeTransaction(transaction, context) {
    const evidence = [];
    let riskScore = 0;
    const anomalies = [];

    // Context destructuring
    const {
        beneficiary,
        entitlement,
        recentTransactions = [],
        shopBaseline = {},
        allBeneficiaries = [],
    } = context;

    // 1. OVER-WITHDRAWAL DETECTION
    if (entitlement) {
        const totalRice = transaction.riceKg;
        const totalWheat = transaction.wheatKg;
        const totalSugar = transaction.sugarKg;

        const riceExcess = totalRice > entitlement.riceKg ? ((totalRice - entitlement.riceKg) / entitlement.riceKg) * 100 : 0;
        const wheatExcess = totalWheat > entitlement.wheatKg ? ((totalWheat - entitlement.wheatKg) / entitlement.wheatKg) * 100 : 0;
        const sugarExcess = totalSugar > entitlement.sugarKg ? ((totalSugar - entitlement.sugarKg) / entitlement.sugarKg) * 100 : 0;

        if (riceExcess > 10 || wheatExcess > 10 || sugarExcess > 10) {
            const excessScore = Math.max(riceExcess, wheatExcess, sugarExcess);
            riskScore += Math.min(excessScore, 40);
            evidence.push(`Over-withdrawal detected: ${excessScore.toFixed(1)}% above entitlement`);
            anomalies.push('OVER_WITHDRAWAL');
        }
    }

    // 2. FREQUENCY ANOMALY (too frequent withdrawals)
    const sameDayTransactions = recentTransactions.filter(t => {
        const tDate = new Date(t.dateTime);
        const txDate = new Date(transaction.dateTime);
        return tDate.toDateString() === txDate.toDateString();
    });

    if (sameDayTransactions.length > 1) {
        riskScore += 25;
        evidence.push(`Multiple withdrawals on same day (${sameDayTransactions.length} transactions)`);
        anomalies.push('HIGH_FREQUENCY');
    }

    // 3. PERIODIC PATTERN DETECTION (suspiciously perfect timing)
    if (recentTransactions.length >= 3) {
        const intervals = [];
        for (let i = 1; i < recentTransactions.length; i++) {
            const diff = new Date(recentTransactions[i].dateTime) - new Date(recentTransactions[i - 1].dateTime);
            intervals.push(diff / (1000 * 60 * 60 * 24)); // days
        }

        const intervalStdDev = stdDev(intervals);
        if (intervalStdDev < 0.5 && intervals.length >= 3) {
            riskScore += 15;
            evidence.push(`Suspiciously periodic withdrawals (std dev: ${intervalStdDev.toFixed(2)} days)`);
            anomalies.push('PERIODIC_PATTERN');
        }
    }

    // 4. AUTHENTICATION FAILURE PATTERN
    if (transaction.authStatus === 'FAILED') {
        riskScore += 20;
        evidence.push('Authentication failed');
        anomalies.push('AUTH_FAILURE');
    }

    const failedAuthCount = recentTransactions.filter(t => t.authStatus === 'FAILED').length;
    if (failedAuthCount >= 2) {
        riskScore += failedAuthCount * 10;
        evidence.push(`Multiple authentication failures (${failedAuthCount} recent failures)`);
        anomalies.push('REPEATED_AUTH_FAILURE');
    }

    // 5. MANUAL OVERRIDE SUSPICION
    if (transaction.authMethod === 'MANUAL') {
        riskScore += 15;
        evidence.push('Manual authentication override used');
        anomalies.push('MANUAL_OVERRIDE');
    }

    // 6. QUANTITY ANOMALY (Z-score based on shop baseline)
    if (shopBaseline.avgQuantity && shopBaseline.quantityStdDev) {
        const totalQty = transaction.riceKg + transaction.wheatKg + transaction.sugarKg;
        const z = Math.abs((totalQty - shopBaseline.avgQuantity) / (shopBaseline.quantityStdDev || 1));

        if (z > 2.5) {
            const anomalyScore = Math.min(z * 5, 25);
            riskScore += anomalyScore;
            evidence.push(`Unusual quantity (${z.toFixed(1)} std deviations from shop average)`);
            anomalies.push('QUANTITY_ANOMALY');
        }
    }

    // 7. GEOGRAPHIC IMPOSSIBILITY (same beneficiary, different shops, short time)
    const recentDifferentShop = recentTransactions.find(t => {
        if (t.fpsId === transaction.fpsId) return false;
        const timeDiff = Math.abs(new Date(transaction.dateTime) - new Date(t.dateTime));
        return timeDiff < 1000 * 60 * 60 * 2; // within 2 hours
    });

    if (recentDifferentShop) {
        riskScore += 30;
        evidence.push('Collected from different shop within 2 hours (geographic impossibility)');
        anomalies.push('GEO_IMPOSSIBLE');
    }

    // 8. DUPLICATE BENEFICIARY DETECTION
    if (allBeneficiaries.length > 0) {
        const similarBeneficiaries = allBeneficiaries.filter(b => {
            if (b.id === beneficiary.id) return false;
            const nameSimilar = levenshteinSimilarity(b.name.toLowerCase(), beneficiary.name.toLowerCase()) > 0.85;
            const ageSimilar = Math.abs(b.age - beneficiary.age) <= 2;
            const addressSimilar = b.address.toLowerCase().includes(beneficiary.address.toLowerCase().split(' ')[0]);
            return nameSimilar && ageSimilar && addressSimilar;
        });

        if (similarBeneficiaries.length > 0) {
            riskScore += 35;
            evidence.push(`Potential duplicate beneficiary (${similarBeneficiaries.length} similar profiles found)`);
            anomalies.push('DUPLICATE_BENEFICIARY');
        }
    }

    // 9. BULK DISTRIBUTION SPIKE
    if (shopBaseline.dailyAvg) {
        const totalQty = transaction.riceKg + transaction.wheatKg + transaction.sugarKg;
        if (totalQty > shopBaseline.dailyAvg * 3) {
            riskScore += 20;
            evidence.push(`Bulk distribution spike (${(totalQty / shopBaseline.dailyAvg).toFixed(1)}x daily average)`);
            anomalies.push('BULK_SPIKE');
        }
    }

    // Calculate severity
    let severity = 'LOW';
    if (riskScore >= 70) severity = 'CRITICAL';
    else if (riskScore >= 50) severity = 'HIGH';
    else if (riskScore >= 30) severity = 'MEDIUM';

    // Recommended action
    let recommendedAction = 'Monitor';
    if (severity === 'CRITICAL') recommendedAction = 'Immediate Investigation Required';
    else if (severity === 'HIGH') recommendedAction = 'Inspector Review Required';
    else if (severity === 'MEDIUM') recommendedAction = 'Flag for Review';

    return {
        riskScore: Math.min(riskScore, 100),
        severity,
        anomalyType: anomalies.join(', ') || 'NONE',
        evidence,
        recommendedAction,
    };
}

/**
 * Analyze FPS shop for stock mismatch and diversion
 */
export async function analyzeShopStock(shopId, stockLogs, transactions) {
    const evidence = [];
    let riskScore = 0;

    // Calculate expected vs actual stock
    const stockIn = {
        rice: stockLogs.filter(s => s.riceIn > 0).reduce((sum, s) => sum + s.riceIn, 0),
        wheat: stockLogs.filter(s => s.wheatIn > 0).reduce((sum, s) => sum + s.wheatIn, 0),
        sugar: stockLogs.filter(s => s.sugarIn > 0).reduce((sum, s) => sum + s.sugarIn, 0),
    };

    const distributed = {
        rice: transactions.reduce((sum, t) => sum + t.riceKg, 0),
        wheat: transactions.reduce((sum, t) => sum + t.wheatKg, 0),
        sugar: transactions.reduce((sum, t) => sum + t.sugarKg, 0),
    };

    const expected = {
        rice: stockIn.rice - distributed.rice,
        wheat: stockIn.wheat - distributed.wheat,
        sugar: stockIn.sugar - distributed.sugar,
    };

    // Stock mismatch detection
    const riceMismatch = Math.abs(expected.rice) > 50 ? (Math.abs(expected.rice) / stockIn.rice) * 100 : 0;
    const wheatMismatch = Math.abs(expected.wheat) > 50 ? (Math.abs(expected.wheat) / stockIn.wheat) * 100 : 0;
    const sugarMismatch = Math.abs(expected.sugar) > 50 ? (Math.abs(expected.sugar) / stockIn.sugar) * 100 : 0;

    const maxMismatch = Math.max(riceMismatch, wheatMismatch, sugarMismatch);

    if (maxMismatch > 15) {
        riskScore += Math.min(maxMismatch * 2, 60);
        evidence.push(`Stock mismatch: ${maxMismatch.toFixed(1)}% discrepancy detected`);
    }

    // Distribution pattern analysis
    const avgDailyDist = transactions.length > 0 ? transactions.length / 30 : 0;
    const recentDist = transactions.filter(t => {
        const daysDiff = (new Date() - new Date(t.dateTime)) / (1000 * 60 * 60 * 24);
        return daysDiff <= 7;
    }).length;

    if (recentDist > avgDailyDist * 7 * 2) {
        riskScore += 25;
        evidence.push(`Sudden distribution spike (${(recentDist / (avgDailyDist * 7)).toFixed(1)}x normal rate)`);
    }

    let severity = 'LOW';
    if (riskScore >= 70) severity = 'CRITICAL';
    else if (riskScore >= 50) severity = 'HIGH';
    else if (riskScore >= 30) severity = 'MEDIUM';

    return {
        riskScore: Math.min(riskScore, 100),
        severity,
        evidence,
        mismatchPercentage: maxMismatch,
    };
}

/**
 * Levenshtein similarity (0-1)
 */
function levenshteinSimilarity(str1, str2) {
    const longer = str1.length > str2.length ? str1 : str2;
    const shorter = str1.length > str2.length ? str2 : str1;

    if (longer.length === 0) return 1.0;

    const distance = levenshteinDistance(longer, shorter);
    return (longer.length - distance) / longer.length;
}

function levenshteinDistance(str1, str2) {
    const matrix = [];

    for (let i = 0; i <= str2.length; i++) {
        matrix[i] = [i];
    }

    for (let j = 0; j <= str1.length; j++) {
        matrix[0][j] = j;
    }

    for (let i = 1; i <= str2.length; i++) {
        for (let j = 1; j <= str1.length; j++) {
            if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
                matrix[i][j] = matrix[i - 1][j - 1];
            } else {
                matrix[i][j] = Math.min(
                    matrix[i - 1][j - 1] + 1,
                    matrix[i][j - 1] + 1,
                    matrix[i - 1][j] + 1
                );
            }
        }
    }

    return matrix[str2.length][str1.length];
}
