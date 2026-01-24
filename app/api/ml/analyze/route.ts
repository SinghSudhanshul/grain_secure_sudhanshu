import { NextResponse } from 'next/server';
import { orchestrator } from '@/lib/ml/orchestrator';

export async function POST(request: Request) {
  try {
    const transaction = await request.json();

    // In a real scenario, we would validate the transaction schema here
    if (!transaction) {
      return NextResponse.json(
        { error: 'Transaction data required' },
        { status: 400 }
      );
    }

    const analysis = await orchestrator.analyzeTransaction(transaction);

    return NextResponse.json({
      transaction_id: transaction.id,
      timestamp: new Date().toISOString(),
      risk_assessment: analysis
    });

  } catch (error: any) {
    console.error('Analysis API Error:', error);
    return NextResponse.json(
      { error: 'Internal Analysis Error', details: error.message },
      { status: 500 }
    );
  }
}
