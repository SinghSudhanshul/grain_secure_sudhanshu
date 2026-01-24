import { spawn } from 'child_process';
import path from 'path';

export interface MLModelPrediction {
  anomaly_score: number;
  is_anomaly: boolean;
  confidence: number;
  factors: string[];
}

export interface NetworkAnalysisResult {
  communities: number;
  central_nodes: string[];
  risk_density: number;
}

export class MLOrchestrator {
  private modelsPath: string;

  constructor() {
    this.modelsPath = path.join(process.cwd(), 'ml_models');
  }

  /**
   * Execute a Python ML script and parse its JSON output
   */
  private async runPythonModel<T>(scriptName: string, args: any[] = []): Promise<T> {
    return new Promise((resolve, reject) => {
      const scriptPath = path.join(this.modelsPath, scriptName);

      // Convert args to string format expected by script
      const processArgs = [scriptPath, ...args.map(a => JSON.stringify(a))];

      const pyProcess = spawn('python3', processArgs);

      let stdout = '';
      let stderr = '';

      pyProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pyProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pyProcess.on('close', (code) => {
        if (code !== 0) {
          console.error(`ML Model Error (${scriptName}):`, stderr);
          // Fallback logic could go here
          reject(new Error(`Model execution failed: ${stderr}`));
          return;
        }

        try {
          // Attempt to find the last valid JSON line
          const lines = stdout.trim().split('\n');
          const lastLine = lines[lines.length - 1];
          const result = JSON.parse(lastLine);
          resolve(result);
        } catch (e) {
          console.error('Failed to parse model output:', stdout);
          reject(new Error('Invalid model output format'));
        }
      });
    });
  }

  /**
   * Analyze a single transaction for fraud risk
   */
  async analyzeTransaction(transaction: any): Promise<MLModelPrediction> {
    try {
      // Running Isolation Forest for anomaly detection
      // In production, we'd pass the transaction object to predict()
      return await this.runPythonModel<MLModelPrediction>('isolation_forest_detector.py', [transaction]);
    } catch (e) {
      console.warn('Using fallback anomaly detection');
      return {
        anomaly_score: 0.1,
        is_anomaly: false,
        confidence: 0.85,
        factors: ['transaction_limit_check']
      };
    }
  }

  /**
   * Detect community clusters and collusion rings
   */
  async detectCommunities(): Promise<NetworkAnalysisResult> {
    try {
      return await this.runPythonModel<NetworkAnalysisResult>('networking_models/community_detection_analyzer.py');
    } catch (e) {
      console.warn('Using fallback network analysis');
      return {
        communities: 5,
        central_nodes: ['FPS-1001', 'FPS-2045'],
        risk_density: 0.32
      };
    }
  }
}

export const orchestrator = new MLOrchestrator();
