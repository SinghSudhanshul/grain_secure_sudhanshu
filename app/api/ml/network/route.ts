import { NextResponse } from 'next/server';
import { orchestrator } from '@/lib/ml/orchestrator';

export async function POST(request: Request) {
  try {
    const { entities, depth } = await request.json();

    const analysis = await orchestrator.detectCommunities();

    return NextResponse.json({
      timestamp: new Date().toISOString(),
      network_health: {
        total_clusters: analysis.communities,
        high_risk_nodes: analysis.central_nodes,
        density_score: analysis.risk_density
      },
      recommendations: [
        "Increase audit frequency for Cluster A",
        "Investigate central node FPS-1001 for collusion"
      ]
    });

  } catch (error: any) {
    console.error('Network API Error:', error);
    return NextResponse.json(
      { error: 'Network Analysis Error', details: error.message },
      { status: 500 }
    );
  }
}
