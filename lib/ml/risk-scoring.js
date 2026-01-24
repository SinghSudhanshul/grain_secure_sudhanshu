/**
 * Risk Scoring Engine
 * Weighted factor model combining multiple risk indicators
 */

/**
 * Weight configuration for risk factors
 */
const DEFAULT_WEIGHTS = {
    anomalyRate: 0.30,      // 30% - Transaction anomaly frequency
    complaintRate: 0.20,    // 20% - Beneficiary complaints
    stockDiscrepancy: 0.20, // 20% - Inventory mismatches
    inspectionDelay: 0.15,  // 15% - Overdue inspections
    networkSuspicion: 0.15  // 15% - Collusion network evidence
};

/**
 * Normalize value to 0-1 range using min-max scaling
 */
function minMaxNormalize(value, min, max) {
    if (max === min) return 0.5;
    return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

/**
 * Normalize using percentile ranking
 */
function percentileNormalize(value, values) {
    const sorted = [...values].sort((a, b) => a - b);
    const index = sorted.findIndex(v => v >= value);
    return index / sorted.length;
}

/**
 * Sigmoid normalization for unbounded metrics
 */
function sigmoidNormalize(value, midpoint = 50, steepness = 0.1) {
    return 1 / (1 + Math.exp(-steepness * (value - midpoint)));
}

/**
 * Calculate shop risk score
 */
export async function calculateShopRiskScore(shop, data = {}) {
    const {
        transactions = [],
        alerts = [],
        complaints = [],
        stockLogs = [],
        lastInspectionDate = null,
        networkScore = 0,
        allShops = []
    } = data;

    const factors = {};
    const contributions = {};

    // Factor 1: Anomaly Rate (30%)
    const anomalousTransactions = transactions.filter(t => t.riskScore >= 50).length;
    const anomalyRate = transactions.length > 0
        ? anomalousTransactions / transactions.length
        : 0;
    
    factors.anomalyRate = Math.min(anomalyRate * 2, 1); // Scale to 0-1
    contributions.anomalyRate = factors.anomalyRate * DEFAULT_WEIGHTS.anomalyRate * 100;

    // Factor 2: Complaint Rate (20%)
    const monthlyComplaintRate = complaints.length / Math.max(transactions.length / 30, 1);
    factors.complaintRate = sigmoidNormalize(monthlyComplaintRate, 0.1, 20);
    contributions.complaintRate = factors.complaintRate * DEFAULT_WEIGHTS.complaintRate * 100;

    // Factor 3: Stock Discrepancy (20%)
    const stockDiscrepancy = calculateStockDiscrepancy(stockLogs, transactions);
    factors.stockDiscrepancy = Math.min(stockDiscrepancy / 20, 1); // Max 20% discrepancy = score 1
    contributions.stockDiscrepancy = factors.stockDiscrepancy * DEFAULT_WEIGHTS.stockDiscrepancy * 100;

    // Factor 4: Inspection Delay (15%)
    const daysSinceInspection = lastInspectionDate
        ? (Date.now() - new Date(lastInspectionDate).getTime()) / (1000 * 60 * 60 * 24)
        : 365;
    
    factors.inspectionDelay = Math.min(daysSinceInspection / 180, 1); // 180 days = score 1
    contributions.inspectionDelay = factors.inspectionDelay * DEFAULT_WEIGHTS.inspectionDelay * 100;

    // Factor 5: Network Suspicion (15%)
    factors.networkSuspicion = Math.min(networkScore, 1);
    contributions.networkSuspicion = factors.networkSuspicion * DEFAULT_WEIGHTS.networkSuspicion * 100;

    // Calculate weighted total (0-100 scale)
    const totalScore = Object.keys(DEFAULT_WEIGHTS).reduce((sum, key) => {
        return sum + (factors[key] * DEFAULT_WEIGHTS[key] * 100);
    }, 0);

    // Determine severity
    let severity = 'LOW';
    if (totalScore >= 70) severity = 'CRITICAL';
    else if (totalScore >= 50) severity = 'HIGH';
    else if (totalScore >= 30) severity = 'MEDIUM';

    // Generate recommendation
    const recommendation = generateShopRecommendation(totalScore, contributions);

    return {
        shopId: shop.id,
        shopName: shop.name,
        riskScore: Math.round(totalScore * 10) / 10,
        severity,
        factors: {
            anomalyRate: Math.round(factors.anomalyRate * 100) / 100,
            complaintRate: Math.round(factors.complaintRate * 100) / 100,
            stockDiscrepancy: Math.round(factors.stockDiscrepancy * 100) / 100,
            inspectionDelay: Math.round(factors.inspectionDelay * 100) / 100,
            networkSuspicion: Math.round(factors.networkSuspicion * 100) / 100
        },
        contributions: {
            anomalyRate: Math.round(contributions.anomalyRate * 10) / 10,
            complaintRate: Math.round(contributions.complaintRate * 10) / 10,
            stockDiscrepancy: Math.round(contributions.stockDiscrepancy * 10) / 10,
            inspectionDelay: Math.round(contributions.inspectionDelay * 10) / 10,
            networkSuspicion: Math.round(contributions.networkSuspicion * 10) / 10
        },
        recommendation,
        metadata: {
            totalTransactions: transactions.length,
            anomalousCount: anomalousTransactions,
            complaintCount: complaints.length,
            daysSinceInspection: Math.round(daysSinceInspection)
        }
    };
}

/**
 * Calculate beneficiary risk score
 */
export async function calculateBeneficiaryRiskScore(beneficiary, data = {}) {
    const {
        transactions = [],
        alerts = [],
        complaints = [],
        entitlement = {},
        networkScore = 0
    } = data;

    const factors = {};
    const contributions = {};

    // Factor 1: Transaction Anomaly Rate
    const anomalousTransactions = transactions.filter(t => t.riskScore >= 50).length;
    const anomalyRate = transactions.length > 0
        ? anomalousTransactions / transactions.length
        : 0;
    
    factors.anomalyRate = Math.min(anomalyRate * 2, 1);
    contributions.anomalyRate = factors.anomalyRate * 0.35 * 100;

    // Factor 2: Over-collection Pattern
    const overCollections = transactions.filter(t => {
        const totalCollected = t.riceKg + t.wheatKg + t.sugarKg;
        const totalEntitled = (entitlement.riceKg || 0) + (entitlement.wheatKg || 0) + (entitlement.sugarKg || 0);
        return totalCollected > totalEntitled * 1.1; // 10% threshold
    }).length;
    
    factors.overCollection = transactions.length > 0 ? overCollections / transactions.length : 0;
    contributions.overCollection = factors.overCollection * 0.25 * 100;

    // Factor 3: Shop Diversity (suspicious if too many shops)
    const uniqueShops = new Set(transactions.map(t => t.fpsId)).size;
    factors.shopDiversity = Math.min(uniqueShops / 5, 1); // 5+ shops = suspicious
    contributions.shopDiversity = factors.shopDiversity * 0.15 * 100;

    // Factor 4: Complaint History
    const complaintRate = complaints.length / Math.max(transactions.length / 10, 1);
    factors.complaintRate = Math.min(complaintRate, 1);
    contributions.complaintRate = factors.complaintRate * 0.10 * 100;

    // Factor 5: Network Suspicion
    factors.networkSuspicion = Math.min(networkScore, 1);
    contributions.networkSuspicion = factors.networkSuspicion * 0.15 * 100;

    // Total score
    const totalScore = Object.values(contributions).reduce((a, b) => a + b, 0);

    // Severity
    let severity = 'LOW';
    if (totalScore >= 70) severity = 'CRITICAL';
    else if (totalScore >= 50) severity = 'HIGH';
    else if (totalScore >= 30) severity = 'MEDIUM';

    const recommendation = generateBeneficiaryRecommendation(totalScore, contributions);

    return {
        beneficiaryId: beneficiary.id,
        beneficiaryName: beneficiary.name,
        riskScore: Math.round(totalScore * 10) / 10,
        severity,
        factors,
        contributions,
        recommendation,
        metadata: {
            totalTransactions: transactions.length,
            anomalousCount: anomalousTransactions,
            uniqueShops,
            complaintCount: complaints.length
        }
    };
}

/**
 * Calculate stock discrepancy percentage
 */
function calculateStockDiscrepancy(stockLogs, transactions) {
    const stockIn = stockLogs.reduce((sum, log) => {
        return sum + (log.riceIn || 0) + (log.wheatIn || 0) + (log.sugarIn || 0);
    }, 0);

    const stockOut = stockLogs.reduce((sum, log) => {
        return sum + (log.riceOut || 0) + (log.wheatOut || 0) + (log.sugarOut || 0);
    }, 0);

    const distributed = transactions.reduce((sum, t) => {
        return sum + t.riceKg + t.wheatKg + t.sugarKg;
    }, 0);

    if (stockIn === 0) return 0;

    const expected = stockIn - distributed;
    const actual = stockIn - stockOut;
    
    return Math.abs((actual - expected) / stockIn) * 100;
}

/**
 * Generate shop-specific recommendation
 */
function generateShopRecommendation(score, contributions) {
    if (score >= 80) {
        const topFactor = Object.keys(contributions).reduce((a, b) =>
            contributions[a] > contributions[b] ? a : b
        );
        return `URGENT: Immediate suspension and investigation required. Primary concern: ${formatFactorName(topFactor)}.`;
    } else if (score >= 60) {
        return 'HIGH PRIORITY: Schedule inspection within 7 days. Review transaction patterns and stock records.';
    } else if (score >= 40) {
        return 'MEDIUM PRIORITY: Include in next routine inspection cycle. Monitor for pattern changes.';
    } else {
        return 'LOW RISK: Continue standard monitoring.';
    }
}

/**
 * Generate beneficiary-specific recommendation
 */
function generateBeneficiaryRecommendation(score, contributions) {
    if (score >= 80) {
        return 'CRITICAL: Investigate for potential fraud. Verify identity and entitlement eligibility.';
    } else if (score >= 60) {
        return 'HIGH: Review transaction history for irregularities. Consider account verification.';
    } else if (score >= 40) {
        return 'MEDIUM: Monitor for unusual patterns. May warrant spot check.';
    } else {
        return 'LOW RISK: Normal beneficiary behavior.';
    }
}

/**
 * Format factor name for display
 */
function formatFactorName(factor) {
    const names = {
        anomalyRate: 'Transaction Anomalies',
        complaintRate: 'Beneficiary Complaints',
        stockDiscrepancy: 'Stock Mismatches',
        inspectionDelay: 'Overdue Inspection',
        networkSuspicion: 'Network Collusion',
        overCollection: 'Excessive Collections',
        shopDiversity: 'Multiple Shop Usage'
    };
    return names[factor] || factor;
}

/**
 * Batch calculate risk scores for multiple entities
 */
export async function calculateBatchRiskScores(entities, type, dataProvider) {
    const results = [];
    
    for (const entity of entities) {
        const data = await dataProvider(entity.id);
        
        const score = type === 'shop'
            ? await calculateShopRiskScore(entity, data)
            : await calculateBeneficiaryRiskScore(entity, data);
        
        results.push(score);
    }
    
    return results;
}

/**
 * Compare risk scores and identify outliers
 */
export function identifyHighRiskEntities(riskScores, threshold = 60) {
    return riskScores
        .filter(score => score.riskScore >= threshold)
        .sort((a, b) => b.riskScore - a.riskScore);
}

export { DEFAULT_WEIGHTS, minMaxNormalize, percentileNormalize, sigmoidNormalize };
