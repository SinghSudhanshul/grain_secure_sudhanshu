#!/usr/bin/env node

/**
 * ML Model Upgrade CLI
 * Run: npm run upgrade-models
 */

const { modelUpgrader } = require('../lib/ai/model-upgrader.ts');
const { writeFileSync } = require('fs');
const { join } = require('path');

async function main() {
  console.log('🤖 Starting ML Model Upgrade Process...\n');

  const modelTypes = ['anomaly-detection', 'risk-scoring', 'network-analysis'];
  const results = [];

  try {
    for (const modelType of modelTypes) {
      console.log(`\n📦 Checking ${modelType}...`);

      const result = await modelUpgrader.monitorAndUpgrade(modelType);

      if (result) {
        results.push(result);
        console.log(`✅ Upgraded ${modelType}:`);
        console.log(`   ${result.previousVersion} → ${result.newVersion}`);
        console.log(`   Performance Improvement: ${result.performanceImprovement.toFixed(2)}%`);
      } else {
        console.log(`✓ ${modelType} performing well - no upgrade needed`);
      }
    }

    // Generate report
    if (results.length > 0) {
      const report = modelUpgrader.generateUpgradeReport(results);
      const reportPath = join(process.cwd(), 'model-upgrade-report.md');
      writeFileSync(reportPath, report);
      console.log(`\n📄 Upgrade report generated: ${reportPath}`);
    }

    console.log('\n✨ Model upgrade process complete!\n');
    process.exit(0);

  } catch (error) {
    console.error('❌ Upgrade failed:', error.message);
    process.exit(1);
  }
}

main();
