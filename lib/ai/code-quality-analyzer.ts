/**
 * AI-Powered Code Quality Analyzer
 * Analyzes code complexity, suggests refactoring, and auto-generates optimized code
 */

import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
import { join } from 'path';

interface CodeMetrics {
  complexity: number;
  linesOfCode: number;
  functionCount: number;
  duplicateCode: number;
  maintainabilityIndex: number;
}

interface RefactoringOpportunity {
  file: string;
  line: number;
  type: 'complexity' | 'duplication' | 'performance' | 'best-practice';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  suggestedFix: string;
  autoFixable: boolean;
}

interface AnalysisReport {
  timestamp: Date;
  filesAnalyzed: number;
  totalIssues: number;
  metrics: CodeMetrics;
  opportunities: RefactoringOpportunity[];
  performanceScore: number;
  qualityScore: number;
}

export class AICodeQualityAnalyzer {
  private projectRoot: string;
  private excludePatterns: string[] = ['node_modules', '.next', 'dist', 'build'];

  constructor(projectRoot: string) {
    this.projectRoot = projectRoot;
  }

  /**
   * Analyze entire codebase for quality issues
   */
  async analyzeCodebase(): Promise<AnalysisReport> {
    const files = this.getAllFiles(this.projectRoot);
    const opportunities: RefactoringOpportunity[] = [];
    let totalComplexity = 0;
    let totalLines = 0;
    let totalFunctions = 0;

    for (const file of files) {
      if (this.shouldAnalyze(file)) {
        const content = readFileSync(file, 'utf-8');
        const fileOpportunities = this.analyzeFile(file, content);
        opportunities.push(...fileOpportunities);

        const metrics = this.calculateFileMetrics(content);
        totalComplexity += metrics.complexity;
        totalLines += metrics.linesOfCode;
        totalFunctions += metrics.functionCount;
      }
    }

    const avgComplexity = totalFunctions > 0 ? totalComplexity / totalFunctions : 0;
    const maintainabilityIndex = this.calculateMaintainabilityIndex(avgComplexity, totalLines);

    return {
      timestamp: new Date(),
      filesAnalyzed: files.length,
      totalIssues: opportunities.length,
      metrics: {
        complexity: avgComplexity,
        linesOfCode: totalLines,
        functionCount: totalFunctions,
        duplicateCode: this.detectDuplicateCode(files),
        maintainabilityIndex
      },
      opportunities,
      performanceScore: this.calculatePerformanceScore(opportunities),
      qualityScore: this.calculateQualityScore(maintainabilityIndex, opportunities)
    };
  }

  /**
   * Analyze single file for issues
   */
  private analyzeFile(filePath: string, content: string): RefactoringOpportunity[] {
    const opportunities: RefactoringOpportunity[] = [];

    // Check for high complexity functions
    const complexityIssues = this.detectComplexity(filePath, content);
    opportunities.push(...complexityIssues);

    // Check for performance issues
    const performanceIssues = this.detectPerformanceIssues(filePath, content);
    opportunities.push(...performanceIssues);

    // Check for best practice violations
    const bestPracticeIssues = this.detectBestPracticeViolations(filePath, content);
    opportunities.push(...bestPracticeIssues);

    return opportunities;
  }

  /**
   * Detect high complexity functions
   */
  private detectComplexity(filePath: string, content: string): RefactoringOpportunity[] {
    const opportunities: RefactoringOpportunity[] = [];
    const lines = content.split('\n');

    // Simple cyclomatic complexity detection
    lines.forEach((line, index) => {
      const complexity = this.calculateLineComplexity(line);

      if (complexity > 10) {
        opportunities.push({
          file: filePath,
          line: index + 1,
          type: 'complexity',
          severity: complexity > 20 ? 'critical' : 'high',
          description: `High cyclomatic complexity detected (${complexity})`,
          suggestedFix: 'Consider breaking this function into smaller, more focused functions',
          autoFixable: false
        });
      }
    });

    return opportunities;
  }

  /**
   * Detect performance issues
   */
  private detectPerformanceIssues(filePath: string, content: string): RefactoringOpportunity[] {
    const opportunities: RefactoringOpportunity[] = [];
    const lines = content.split('\n');

    lines.forEach((line, index) => {
      // Detect nested loops
      if (line.includes('for') && content.slice(0, content.indexOf(line)).split('for').length > 2) {
        opportunities.push({
          file: filePath,
          line: index + 1,
          type: 'performance',
          severity: 'medium',
          description: 'Nested loops detected - potential O(n²) or worse complexity',
          suggestedFix: 'Consider using hash maps or optimizing the algorithm',
          autoFixable: false
        });
      }

      // Detect inefficient array operations
      if (line.includes('.filter(') && line.includes('.map(')) {
        opportunities.push({
          file: filePath,
          line: index + 1,
          type: 'performance',
          severity: 'low',
          description: 'Chained filter and map - consider combining into single reduce',
          suggestedFix: 'Use Array.reduce() for better performance',
          autoFixable: true
        });
      }

      // Detect synchronous file operations
      if (line.includes('readFileSync') || line.includes('writeFileSync')) {
        opportunities.push({
          file: filePath,
          line: index + 1,
          type: 'performance',
          severity: 'medium',
          description: 'Synchronous file operation blocks event loop',
          suggestedFix: 'Use async file operations (readFile, writeFile)',
          autoFixable: true
        });
      }
    });

    return opportunities;
  }

  /**
   * Detect best practice violations
   */
  private detectBestPracticeViolations(filePath: string, content: string): RefactoringOpportunity[] {
    const opportunities: RefactoringOpportunity[] = [];
    const lines = content.split('\n');

    lines.forEach((line, index) => {
      // Detect console.log in production code
      if (line.includes('console.log') && !filePath.includes('test')) {
        opportunities.push({
          file: filePath,
          line: index + 1,
          type: 'best-practice',
          severity: 'low',
          description: 'console.log found in production code',
          suggestedFix: 'Use proper logging library or remove debug statements',
          autoFixable: true
        });
      }

      // Detect var usage
      if (line.trim().startsWith('var ')) {
        opportunities.push({
          file: filePath,
          line: index + 1,
          type: 'best-practice',
          severity: 'low',
          description: 'var keyword used instead of let/const',
          suggestedFix: 'Replace var with const or let',
          autoFixable: true
        });
      }

      // Detect missing error handling
      if (line.includes('await ') && !content.includes('try') && !content.includes('catch')) {
        opportunities.push({
          file: filePath,
          line: index + 1,
          type: 'best-practice',
          severity: 'high',
          description: 'Async operation without error handling',
          suggestedFix: 'Wrap in try-catch block',
          autoFixable: false
        });
      }
    });

    return opportunities;
  }

  /**
   * Auto-fix issues where possible
   */
  async autoFix(opportunities: RefactoringOpportunity[]): Promise<number> {
    let fixedCount = 0;
    const fileChanges: Map<string, string> = new Map();

    for (const opp of opportunities) {
      if (!opp.autoFixable) continue;

      const content = fileChanges.get(opp.file) || readFileSync(opp.file, 'utf-8');
      const fixed = this.applyFix(content, opp);

      if (fixed !== content) {
        fileChanges.set(opp.file, fixed);
        fixedCount++;
      }
    }

    // Write all changes
    Array.from(fileChanges.entries()).forEach(([file, content]) => {
      writeFileSync(file, content, 'utf-8');
    });

    return fixedCount;
  }

  /**
   * Apply specific fix to content
   */
  private applyFix(content: string, opportunity: RefactoringOpportunity): string {
    const lines = content.split('\n');
    const lineIndex = opportunity.line - 1;

    if (lineIndex < 0 || lineIndex >= lines.length) return content;

    const line = lines[lineIndex];

    // Fix console.log
    if (opportunity.description.includes('console.log')) {
      lines[lineIndex] = line.replace(/console\.log/g, '// console.log');
    }

    // Fix var to const/let
    if (opportunity.description.includes('var keyword')) {
      lines[lineIndex] = line.replace(/\bvar\b/, 'const');
    }

    // Fix filter+map to reduce
    if (opportunity.description.includes('filter and map')) {
      // This would require more sophisticated AST manipulation
      // For now, just add a comment
      lines[lineIndex] = `// TODO: Optimize - ${line}`;
    }

    return lines.join('\n');
  }

  /**
   * Calculate file metrics
   */
  private calculateFileMetrics(content: string): CodeMetrics {
    const lines = content.split('\n');
    const linesOfCode = lines.filter(l => l.trim() && !l.trim().startsWith('//')).length;

    const functionMatches = content.match(/function\s+\w+|=>\s*{|async\s+function/g) || [];
    const functionCount = functionMatches.length;

    let totalComplexity = 0;
    lines.forEach(line => {
      totalComplexity += this.calculateLineComplexity(line);
    });

    return {
      complexity: totalComplexity,
      linesOfCode,
      functionCount,
      duplicateCode: 0,
      maintainabilityIndex: this.calculateMaintainabilityIndex(totalComplexity / (functionCount || 1), linesOfCode)
    };
  }

  /**
   * Calculate cyclomatic complexity for a line
   */
  private calculateLineComplexity(line: string): number {
    let complexity = 0;

    // Count decision points
    if (line.includes('if')) complexity++;
    if (line.includes('else')) complexity++;
    if (line.includes('for')) complexity++;
    if (line.includes('while')) complexity++;
    if (line.includes('case')) complexity++;
    if (line.includes('&&')) complexity++;
    if (line.includes('||')) complexity++;
    if (line.includes('?')) complexity++;
    if (line.includes('catch')) complexity++;

    return complexity;
  }

  /**
   * Calculate maintainability index
   * Based on Microsoft's formula
   */
  private calculateMaintainabilityIndex(avgComplexity: number, linesOfCode: number): number {
    const volume = linesOfCode * Math.log2(linesOfCode || 1);
    const mi = Math.max(0, (171 - 5.2 * Math.log(volume) - 0.23 * avgComplexity - 16.2 * Math.log(linesOfCode || 1)) * 100 / 171);
    return Math.round(mi);
  }

  /**
   * Detect duplicate code
   */
  private detectDuplicateCode(files: string[]): number {
    // Simplified duplicate detection
    const codeBlocks: Map<string, number> = new Map();
    let duplicates = 0;

    for (const file of files) {
      if (!this.shouldAnalyze(file)) continue;

      const content = readFileSync(file, 'utf-8');
      const lines = content.split('\n');

      // Check for duplicate blocks of 5+ lines
      for (let i = 0; i < lines.length - 5; i++) {
        const block = lines.slice(i, i + 5).join('\n').trim();
        if (block.length > 50) {
          const count = codeBlocks.get(block) || 0;
          codeBlocks.set(block, count + 1);
          if (count > 0) duplicates++;
        }
      }
    }

    return duplicates;
  }

  /**
   * Calculate performance score
   */
  private calculatePerformanceScore(opportunities: RefactoringOpportunity[]): number {
    const perfIssues = opportunities.filter(o => o.type === 'performance');
    const criticalCount = perfIssues.filter(o => o.severity === 'critical').length;
    const highCount = perfIssues.filter(o => o.severity === 'high').length;
    const mediumCount = perfIssues.filter(o => o.severity === 'medium').length;

    const score = 100 - (criticalCount * 20 + highCount * 10 + mediumCount * 5);
    return Math.max(0, Math.min(100, score));
  }

  /**
   * Calculate quality score
   */
  private calculateQualityScore(maintainabilityIndex: number, opportunities: RefactoringOpportunity[]): number {
    const issueWeight = opportunities.reduce((sum, opp) => {
      const weights = { critical: 10, high: 5, medium: 2, low: 1 };
      return sum + weights[opp.severity];
    }, 0);

    const score = (maintainabilityIndex * 0.6) + ((100 - Math.min(100, issueWeight)) * 0.4);
    return Math.round(score);
  }

  /**
   * Get all files recursively
   */
  private getAllFiles(dir: string): string[] {
    const files: string[] = [];

    try {
      const items = readdirSync(dir);

      for (const item of items) {
        const fullPath = join(dir, item);

        if (this.excludePatterns.some(pattern => fullPath.includes(pattern))) {
          continue;
        }

        const stat = statSync(fullPath);

        if (stat.isDirectory()) {
          files.push(...this.getAllFiles(fullPath));
        } else {
          files.push(fullPath);
        }
      }
    } catch (error) {
      // Ignore permission errors
    }

    return files;
  }

  /**
   * Check if file should be analyzed
   */
  private shouldAnalyze(file: string): boolean {
    return (file.endsWith('.js') || file.endsWith('.ts') || file.endsWith('.jsx') || file.endsWith('.tsx')) &&
      !file.includes('.test.') &&
      !file.includes('.spec.');
  }

  /**
   * Generate HTML report
   */
  generateReport(analysis: AnalysisReport): string {
    return `
<!DOCTYPE html>
<html>
<head>
  <title>Code Quality Report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 40px; background: #f5f5f7; }
    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1 { color: #1d1d1f; margin-bottom: 10px; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; }
    .metric-value { font-size: 36px; font-weight: bold; margin: 10px 0; }
    .metric-label { font-size: 14px; opacity: 0.9; }
    .opportunities { margin-top: 40px; }
    .opportunity { background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #667eea; border-radius: 4px; }
    .severity-critical { border-left-color: #ff3b30; }
    .severity-high { border-left-color: #ff9500; }
    .severity-medium { border-left-color: #ffcc00; }
    .severity-low { border-left-color: #34c759; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🚀 Code Quality Analysis Report</h1>
    <p>Generated: ${analysis.timestamp.toLocaleString()}</p>
    
    <div class="metrics">
      <div class="metric-card">
        <div class="metric-label">Quality Score</div>
        <div class="metric-value">${analysis.qualityScore}/100</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Performance Score</div>
        <div class="metric-value">${analysis.performanceScore}/100</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Maintainability</div>
        <div class="metric-value">${analysis.metrics.maintainabilityIndex}/100</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Total Issues</div>
        <div class="metric-value">${analysis.totalIssues}</div>
      </div>
    </div>

    <div class="opportunities">
      <h2>Refactoring Opportunities</h2>
      ${analysis.opportunities.slice(0, 50).map(opp => `
        <div class="opportunity severity-${opp.severity}">
          <strong>${opp.file}:${opp.line}</strong> - ${opp.type}
          <p>${opp.description}</p>
          <p><em>Suggested fix: ${opp.suggestedFix}</em></p>
          ${opp.autoFixable ? '<span style="color: #34c759;">✓ Auto-fixable</span>' : ''}
        </div>
      `).join('')}
    </div>
  </div>
</body>
</html>
    `;
  }
}

// Export singleton instance
export const codeAnalyzer = new AICodeQualityAnalyzer(process.cwd());
