/**
 * Isolation Forest Algorithm for Anomaly Detection
 * Identifies outliers through recursive random partitioning
 */

class IsolationTree {
    constructor(maxDepth = 10) {
        this.maxDepth = maxDepth;
        this.root = null;
    }

    /**
     * Build isolation tree from sample data
     */
    fit(data, depth = 0) {
        if (depth >= this.maxDepth || data.length <= 1) {
            return { size: data.length, isLeaf: true };
        }

        // Randomly select a feature
        const features = Object.keys(data[0]);
        const feature = features[Math.floor(Math.random() * features.length)];

        // Get min and max values for the selected feature
        const values = data.map(d => d[feature]);
        const min = Math.min(...values);
        const max = Math.max(...values);

        if (min === max) {
            return { size: data.length, isLeaf: true };
        }

        // Random split point
        const splitValue = min + Math.random() * (max - min);

        // Split data
        const leftData = data.filter(d => d[feature] < splitValue);
        const rightData = data.filter(d => d[feature] >= splitValue);

        return {
            feature,
            splitValue,
            isLeaf: false,
            left: this.fit(leftData, depth + 1),
            right: this.fit(rightData, depth + 1)
        };
    }

    /**
     * Calculate path length for a data point
     */
    pathLength(point, node = this.root, depth = 0) {
        if (node.isLeaf) {
            // Adjust for leaf size
            return depth + this.c(node.size);
        }

        if (point[node.feature] < node.splitValue) {
            return this.pathLength(point, node.left, depth + 1);
        } else {
            return this.pathLength(point, node.right, depth + 1);
        }
    }

    /**
     * Average path length of unsuccessful search in BST
     */
    c(n) {
        if (n <= 1) return 0;
        const H = Math.log(n - 1) + 0.5772156649; // Euler constant
        return 2 * H - (2 * (n - 1) / n);
    }
}

class IsolationForest {
    constructor(numTrees = 100, sampleSize = 256, maxDepth = 10) {
        this.numTrees = numTrees;
        this.sampleSize = sampleSize;
        this.maxDepth = maxDepth;
        this.trees = [];
        this.avgPathLength = 0;
    }

    /**
     * Train isolation forest on data
     */
    fit(data) {
        this.trees = [];
        
        for (let i = 0; i < this.numTrees; i++) {
            // Random subsample
            const sample = this.randomSample(data, Math.min(this.sampleSize, data.length));
            
            const tree = new IsolationTree(this.maxDepth);
            tree.root = tree.fit(sample);
            this.trees.push(tree);
        }

        // Calculate average path length for normalization
        this.avgPathLength = this.trees[0].c(this.sampleSize);
    }

    /**
     * Random sampling without replacement
     */
    randomSample(data, size) {
        const shuffled = [...data].sort(() => 0.5 - Math.random());
        return shuffled.slice(0, size);
    }

    /**
     * Predict anomaly score for a data point (0 to 1)
     * Higher scores indicate more anomalous
     */
    predict(point) {
        const pathLengths = this.trees.map(tree => tree.pathLength(point));
        const avgPathLength = pathLengths.reduce((a, b) => a + b, 0) / pathLengths.length;
        
        // Anomaly score: 2^(-avgPathLength / c(n))
        // Maps to 0-1 range where 1 is most anomalous
        const anomalyScore = Math.pow(2, -avgPathLength / this.avgPathLength);
        
        return Math.min(Math.max(anomalyScore, 0), 1);
    }

    /**
     * Predict anomaly scores for multiple points
     */
    predictBatch(points) {
        return points.map(point => ({
            point,
            anomalyScore: this.predict(point)
        }));
    }
}

/**
 * Feature extraction from transaction data
 */
function extractFeatures(transaction, context = {}) {
    const {
        recentTransactions = [],
        shopBaseline = {},
        entitlement = {}
    } = context;

    // Calculate derived features
    const totalQty = transaction.riceKg + transaction.wheatKg + transaction.sugarKg;
    
    // Time-based features
    const daysSinceLastTx = recentTransactions.length > 0
        ? (new Date(transaction.dateTime) - new Date(recentTransactions[0].dateTime)) / (1000 * 60 * 60 * 24)
        : 30;

    // Quantity ratios
    const riceRatio = entitlement.riceKg ? transaction.riceKg / entitlement.riceKg : 1;
    const wheatRatio = entitlement.wheatKg ? transaction.wheatKg / entitlement.wheatKg : 1;
    const sugarRatio = entitlement.sugarKg ? transaction.sugarKg / entitlement.sugarKg : 1;

    // Authentication features
    const authMethodScore = transaction.authMethod === 'FACE' ? 2 : transaction.authMethod === 'OTP' ? 1 : 0;
    const authStatusScore = transaction.authStatus === 'SUCCESS' ? 1 : 0;

    // Shop comparison
    const shopDeviation = shopBaseline.avgQuantity
        ? Math.abs(totalQty - shopBaseline.avgQuantity) / (shopBaseline.quantityStdDev || 1)
        : 0;

    // Transaction frequency
    const recentTxCount = recentTransactions.filter(t => {
        const diff = Math.abs(new Date(transaction.dateTime) - new Date(t.dateTime));
        return diff < 1000 * 60 * 60 * 24 * 7; // Last 7 days
    }).length;

    return {
        totalQty,
        daysSinceLastTx,
        riceRatio,
        wheatRatio,
        sugarRatio,
        authMethodScore,
        authStatusScore,
        shopDeviation,
        recentTxCount,
        hour: new Date(transaction.dateTime).getHours(),
        dayOfWeek: new Date(transaction.dateTime).getDay()
    };
}

/**
 * Analyze transactions using Isolation Forest
 */
export async function analyzeWithIsolationForest(transactions, contextMap = {}) {
    if (transactions.length === 0) {
        return [];
    }

    // Extract features for all transactions
    const features = transactions.map((tx, idx) => ({
        id: tx.id,
        features: extractFeatures(tx, contextMap[tx.id] || {}),
        transaction: tx
    }));

    // Train model
    const forest = new IsolationForest(100, 256, 10);
    forest.fit(features.map(f => f.features));

    // Predict anomalies
    const results = features.map(item => {
        const score = forest.predict(item.features);
        
        return {
            transactionId: item.id,
            anomalyScore: score,
            isAnomaly: score > 0.65, // Threshold for anomaly classification
            confidence: score,
            method: 'ISOLATION_FOREST',
            features: item.features
        };
    });

    return results;
}

/**
 * Train and save model for later use
 */
export function trainIsolationForest(trainingData) {
    const features = trainingData.map(tx => extractFeatures(tx, {}));
    
    const forest = new IsolationForest(100, 256, 10);
    forest.fit(features);
    
    return {
        model: forest,
        featureStats: calculateFeatureStatistics(features)
    };
}

/**
 * Calculate feature statistics for normalization
 */
function calculateFeatureStatistics(features) {
    const stats = {};
    const featureKeys = Object.keys(features[0]);
    
    featureKeys.forEach(key => {
        const values = features.map(f => f[key]);
        stats[key] = {
            min: Math.min(...values),
            max: Math.max(...values),
            mean: values.reduce((a, b) => a + b, 0) / values.length,
            stdDev: Math.sqrt(values.map(v => Math.pow(v - stats[key]?.mean || 0, 2)).reduce((a, b) => a + b, 0) / values.length)
        };
    });
    
    return stats;
}

export { IsolationForest, IsolationTree, extractFeatures };
