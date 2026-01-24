// Socket.IO is not compatible with Next.js App Router API routes
// For real-time features, use Server-Sent Events or a separate WebSocket server

import { NextResponse } from 'next/server';

export async function GET(request) {
    return NextResponse.json({
        message: 'Socket.IO endpoint - use separate WebSocket server for real-time features',
        status: 'inactive'
    });
}
