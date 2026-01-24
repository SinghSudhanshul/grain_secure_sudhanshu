#!/usr/bin/env node

/**
 * AI Code Quality Analysis CLI
 * Run: npm run analyze-code
 */

const { codeAnalyzer } = require('../lib/ai/code-quality-analyzer.ts');
const { writeFileSync } = require('fs');
const { join } = require('path');

async function main() {
  console.log('🚀 Starting AI Code Quality Analysis...\n');

  try {
    // Run analysis
    const analysis = await codeAnalyzer.analyzeCodebase();

    // Display summary
    console.log('📊 Analysis Complete!\n');
    console.log(`Files Analyzed: ${analysis.filesAnalyzed}`);
    console.log(`Total Issues: ${analysis.totalIssues}`);
    console.log(`Quality Score: ${analysis.qualityScore}/100`);
    console.log(`Performance Score: ${analysis.performanceScore}/100`);
    console.log(`Maintainability Index: ${analysis.metrics.maintainabilityIndex}/100`);
    console.log(`\nLines of Code: ${analysis.metrics.linesOfCode}`);
    console.log(`Functions: ${analysis.metrics.functionCount}`);
    console.log(`Avg Complexity: ${analysis.metrics.complexity.toFixed(2)}`);
    console.log(`Duplicate Code Blocks: ${analysis.metrics.duplicateCode}`);

    // Show top issues
    console.log(`\n🔍 Top Issues:\n`);
    const criticalIssues = analysis.opportunities
      .filter(o => o.severity === 'critical')
      .slice(0, 5);

    criticalIssues.forEach((issue, i) => {
      console.log(`${i + 1}. [${issue.severity.toUpperCase()}] ${issue.file}:${issue.line}`);
      console.log(`   ${issue.description}`);
      console.log(`   💡 ${issue.suggestedFix}\n`);
    });

    // Auto-fix if requested
    if (process.argv.includes('--fix')) {
      console.log('🔧 Applying auto-fixes...\n');
      const fixedCount = await codeAnalyzer.autoFix(analysis.opportunities);
      console.log(`✅ Fixed ${fixedCount} issues automatically\n`);
    }

    // Generate HTML report
    const report = codeAnalyzer.generateReport(analysis);
    const reportPath = join(process.cwd(), 'code-quality-report.html');
    writeFileSync(reportPath, report);
    console.log(`📄 Full report generated: ${reportPath}\n`);

    // Exit with appropriate code
    process.exit(analysis.qualityScore < 70 ? 1 : 0);

  } catch (error) {
    console.error('❌ Analysis failed:', error.message);
    process.exit(1);
  }
}

main();
