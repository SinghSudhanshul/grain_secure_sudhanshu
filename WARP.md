# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

GrainSecure is an AI-enabled Public Distribution System (PDS) monitoring platform for India's food distribution network. The system detects fraud patterns in real-time, manages investigation cases, provides cryptographic audit trails, and offers public transparency through a Next.js 14 application with JavaScript (not TypeScript).

**Key Distinction**: This implementation uses **JavaScript with JSX** (not TypeScript), SQLite for development (Postgres-ready), and SHA-256 hashing (not bcrypt) for passwords.

## Development Commands

### Essential Commands
```bash
# Install dependencies
npm install

# Database setup (required after clone)
npx prisma generate
npx prisma db push

# Seed database with demo data (400 beneficiaries, 20 shops, 10k transactions)
npm run seed

# Development server (runs on http://localhost:3000)
npm run dev

# Production build
npm run build

# Start production server
npm start

# Linting
npm run lint
```

### Database Operations
```bash
# Regenerate Prisma client after schema changes
npx prisma generate

# Push schema changes to database (dev only)
npx prisma db push

# View database in Prisma Studio
npx prisma studio

# Create new migration (when moving to production Postgres)
npx prisma migrate dev --name description_of_changes
```

## High-Level Architecture

### Core System Design

**Digital Twin Simulator Architecture**
- Real-time transaction generator creates events every 2-5 seconds via `app/api/simulator/control/route.js`
- Socket.IO broadcasts to all connected clients for live dashboard updates
- Admin can inject fraud scenarios (stock diversion, ghost beneficiaries, bulk spikes)
- Each generated transaction runs through AI fraud detection pipeline immediately
- High-risk transactions (score ≥50) automatically create alerts; critical alerts (score ≥70) auto-create investigation cases

**AI Fraud Detection Pipeline** (`lib/fraud-detection.js`)
The fraud detection engine implements 9 distinct detection patterns using statistical methods:
1. **Over-withdrawal**: Compares transaction quantities against monthly entitlements using percentage-based thresholds
2. **High frequency**: Detects multiple same-day transactions suggesting credential sharing
3. **Periodic patterns**: Identifies suspiciously regular withdrawal intervals (low standard deviation < 0.5 days)
4. **Authentication failures**: Tracks failed and repeated auth attempts
5. **Manual override suspicion**: Flags manual authentication bypasses
6. **Quantity anomalies**: Uses z-scores against shop baseline distributions to detect outliers (threshold: 2.5σ)
7. **Geographic impossibility**: Detects collections from different shops within 2-hour windows
8. **Duplicate beneficiaries**: Applies Levenshtein similarity (>0.85) to detect potential duplicate enrollments
9. **Bulk distribution spikes**: Identifies transactions exceeding 3x shop daily average

All detection algorithms return explainable results with evidence arrays, risk scores (0-100), severity levels (LOW/MEDIUM/HIGH/CRITICAL), and recommended actions.

**Cryptographic Audit Chain** (`lib/hash.js`, `AuditLog` model)
- Each audit log contains `prevHash` and `currentHash = SHA256(prevHash + eventType + metaJson + timestamp)`
- Chain integrity verification traverses entire log sequence checking hash continuity
- Any tampering breaks the chain immediately, detectable by auditor dashboard
- Genesis log uses predetermined seed hash; all subsequent logs chain cryptographically

**Case Management Workflow**
Alert lifecycle: `OPEN` → `ASSIGNED` (to inspector) → `INVESTIGATING` → `RESOLVED`
Verdict options: `FRAUD_CONFIRMED`, `FALSE_POSITIVE`, `NEED_MORE_INFO`
Each status change creates an `InspectorAction` record for complete audit trail

### Role-Based Dashboard Architecture

**6 Distinct User Roles** (enforced via `middleware.js` using NextAuth):
1. **ADMIN**: Real-time KPIs, fraud heatmaps, simulator controls (`/admin/dashboard`)
2. **INSPECTOR**: Assigned cases, investigation tools, verdict submission (`/inspector/dashboard`)
3. **DEALER**: Beneficiary verification, stock management, distribution records (`/dealer/dashboard`)
4. **AUDITOR**: Audit log verification, hash chain integrity checks (`/auditor/dashboard`)
5. **BENEFICIARY**: Transaction history, entitlement tracking, dispute filing (`/beneficiary/portal`)
6. **PUBLIC**: Transparency dashboard with aggregated data, no login required (`/public/dashboard`)

Middleware protects all role-specific routes; unauthorized access redirects to login.

### Data Model Relationships

**Core Entities**:
- `District` → contains `FPSShop[]` and `Beneficiary[]`
- `FPSShop` → has `Transaction[]`, `StockLog[]`, `Alert[]`, tracks `riskScore` (0-100)
- `Beneficiary` → has `Entitlement[]` (monthly, commodity-specific), `Transaction[]`, `Alert[]`
- `Transaction` → links beneficiary + shop, contains commodity quantities (rice/wheat/sugar in kg), authentication method (OTP/FACE/MANUAL), includes `riskScore`, `anomalyType`, `anomalyFlags` (JSON evidence)
- `Alert` → references `Transaction`/`Beneficiary`/`FPSShop`, has severity (LOW/MEDIUM/HIGH/CRITICAL), status (OPEN/RESOLVED), evidence JSON
- `Case` → wraps `Alert`, tracks investigation lifecycle, stores inspector assignments, notes, verdict
- `StockLog` → tracks inventory in/out movements per shop for reconciliation

**Key Pattern**: Transactions drive fraud detection → Alerts → Cases → Inspector Actions (complete investigation trail)

### Authentication & Security

**NextAuth Configuration** (`app/api/auth/[...nextauth]/route.js`):
- Credentials provider with email/password
- JWT tokens stored in HTTP-only cookies
- Password hashing: SHA-256 (see `lib/hash.js`), NOT bcrypt
- Session includes user role, id, email for authorization checks
- No API key required for local development

**Authorization Pattern**:
```javascript
// In API routes, verify role:
import { getServerSession } from 'next-auth';
import { authOptions } from '@/app/api/auth/[...nextauth]/route';

const session = await getServerSession(authOptions);
if (!session || session.user.role !== 'ADMIN') {
  return NextResponse.json({ error: 'Unauthorized' }, { status: 403 });
}
```

## API Route Organization

All API routes follow Next.js 14 App Router conventions at `app/api/*/route.js`:

**Simulator Routes** (ADMIN only):
- `POST /api/simulator/control` - Start/stop transaction generator
- `POST /api/simulator/inject-fraud` - Inject specific fraud patterns (stock diversion, ghost beneficiary, bulk spike)

**Role-Specific Dashboards**:
- `GET /api/admin/dashboard` - KPI aggregations, fraud statistics
- `GET /api/inspector/cases` - Assigned cases with filtering
- `GET /api/dealer/dashboard` - Shop inventory and transactions
- `GET /api/auditor/audit-logs` - Hash chain verification endpoint
- `GET /api/beneficiary/portal` - Personal transaction history
- `GET /api/public/dashboard` - Aggregated transparency data (no auth required)

**Real-Time Communication**:
- `/api/socket/route.js` - Socket.IO server exports `broadcastTransaction()`, `broadcastAlert()`, `broadcastSimulatorStatus()`

## Project-Specific Development Patterns

### Working with the Fraud Detection Engine

When modifying detection algorithms in `lib/fraud-detection.js`:

1. **Context Requirements**: `analyzeTransaction()` requires specific context object:
   ```javascript
   {
     beneficiary: { id, name, age, familySize, ... },
     entitlement: { riceKg, wheatKg, sugarKg },
     recentTransactions: [], // last 10 transactions for pattern analysis
     shopBaseline: { avgQuantity, quantityStdDev, dailyAvg },
     allBeneficiaries: [] // for duplicate detection (expensive)
   }
   ```

2. **Return Format**: Always return `{ riskScore, severity, anomalyType, evidence[], recommendedAction }`

3. **Evidence Array**: Must contain human-readable strings explaining WHY each pattern triggered (e.g., "Over-withdrawal detected: 35.2% above entitlement")

4. **Z-Score Calculation**: Uses population standard deviation; handle division by zero (return 0 if stdDev === 0)

5. **Testing Pattern Detection**: Seed data includes ~5% fraudulent transactions; verify new patterns detect these without excessive false positives

### Socket.IO Integration Pattern

Real-time updates require importing broadcast functions:

```javascript
import { broadcastTransaction, broadcastAlert } from '@/app/api/socket/route';

// After creating transaction
broadcastTransaction({
  beneficiaryName: beneficiary.name,
  shopName: shop.name,
  dateTime: transaction.dateTime,
  riceKg, wheatKg, sugarKg
});

// After creating high-risk alert
broadcastAlert({
  title: alert.title,
  description: alert.description,
  severity: alert.severity,
  createdAt: alert.createdAt
});
```

Client-side components must initialize Socket.IO client and listen for events (`transaction`, `alert`, `simulator-status`).

### Database Seeding Strategy

`scripts/seed.js` creates deterministic demo data:
- 1 district (Bangalore Urban)
- 20 FPS shops with realistic lat/lng coordinates for mapping
- 400 beneficiaries with Indian names from predefined lists
- 6 months of entitlements per beneficiary
- 10,000+ historical transactions (~5% with fraud patterns)
- 100 audit logs with valid hash chain
- 5 demo users (one per role) with credentials documented in README.md

**Seeding is idempotent**: Deletes all existing data before recreating; safe to run repeatedly.

### Password Hashing (CRITICAL)

This project uses **SHA-256**, NOT bcrypt:

```javascript
// Correct pattern (see lib/hash.js):
import { hashPassword, verifyPassword } from '@/lib/hash';

const hash = hashPassword('plaintext'); // SHA-256
const isValid = verifyPassword('plaintext', hash); // boolean
```

**Do not** import bcrypt or use bcrypt patterns. The seed script and auth routes both use SHA-256 exclusively.

### Prisma Client Access

Always import from singleton instance to prevent connection exhaustion:

```javascript
import { prisma } from '@/lib/prisma';

// NOT: import { PrismaClient } from '@prisma/client'; const prisma = new PrismaClient();
```

The singleton pattern (`lib/prisma.js`) ensures single client instance across hot reloads during development.

## Common Development Workflows

### Adding a New Fraud Detection Pattern

1. Open `lib/fraud-detection.js`
2. Add detection logic to `analyzeTransaction()` function
3. Calculate pattern-specific risk score contribution (0-40 points typical)
4. Add to `evidence[]` array with explanation
5. Add pattern name to `anomalies[]` array
6. Update aggregate `riskScore` (cap at 100)
7. Test with seed data: `npm run seed && npm run dev`
8. Verify alerts generate for high-risk transactions in admin dashboard

### Creating New API Endpoint

1. Create `app/api/[route-name]/route.js`
2. Export HTTP method handlers: `export async function GET(request) { ... }`
3. Add authentication check using `getServerSession(authOptions)`
4. Implement role-based authorization if needed
5. Use Prisma client from `@/lib/prisma` for database operations
6. Return `NextResponse.json()` for success/error responses
7. Add to middleware matcher if route requires protection

### Modifying Database Schema

1. Edit `prisma/schema.prisma`
2. Run `npx prisma generate` to update client types
3. Run `npx prisma db push` to apply changes to SQLite (dev)
4. Update seed script if new fields require default values
5. Re-run `npm run seed` to populate new fields
6. For production Postgres, create migration: `npx prisma migrate dev`

### Testing Fraud Detection

1. Start simulator: Admin dashboard → "Start Simulation" button
2. Inject fraud: Click "💣 Inject: [Fraud Type]" buttons
3. Observe alert generation in real-time feed
4. Check Cases page for auto-created critical cases
5. Verify evidence explanations make sense
6. Test false positive rate with normal transactions

### Debugging Real-Time Features

1. Check browser console for Socket.IO connection logs
2. Verify server logs show Socket.IO initialization
3. Test multiple browser windows to confirm broadcast
4. Use `broadcastSimulatorStatus()` to debug simulator state
5. Check network tab for Socket.IO upgrade to WebSocket protocol

## Technology Constraints & Decisions

**Language**: JavaScript with JSX (NOT TypeScript) - respect existing patterns, avoid type annotations

**Database**: SQLite for development (`prisma/dev.db`), Postgres-ready via Prisma schema changes

**Authentication**: NextAuth v4 with credentials provider, SHA-256 password hashing

**Real-Time**: Socket.IO for server-push updates (transactions, alerts, simulator status)

**Styling**: TailwindCSS + shadcn/ui components (located in `components/ui/`)

**Charts**: Recharts for analytics dashboards

**Maps**: Leaflet + OpenStreetMap for shop location visualization and fraud heatmaps

**No External ML Services**: All fraud detection runs in-process using pure JavaScript statistical methods

## Demo Credentials

These are documented in README.md and created by seed script:

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@grainsecure.in | admin123 |
| Inspector | inspector@grainsecure.in | inspector123 |
| Dealer | dealer@grainsecure.in | dealer123 |
| Auditor | auditor@grainsecure.in | auditor123 |
| Beneficiary | beneficiary@grainsecure.in | beneficiary123 |

## Critical Files for Understanding System

1. **`lib/fraud-detection.js`** - Core AI engine with 9 detection patterns
2. **`app/api/simulator/control/route.js`** - Digital twin transaction generator
3. **`app/api/socket/route.js`** - Real-time broadcast infrastructure  
4. **`prisma/schema.prisma`** - Complete data model
5. **`scripts/seed.js`** - Demo data generation logic
6. **`lib/hash.js`** - Cryptographic utilities for audit chain
7. **`middleware.js`** - Route protection and role-based access control

## Migration Path to Production

When moving from development (SQLite) to production (Postgres):

1. Update `prisma/schema.prisma` datasource:
   ```prisma
   datasource db {
     provider = "postgresql"
     url      = env("DATABASE_URL")
   }
   ```

2. Create initial migration: `npx prisma migrate dev --name init`

3. Set `DATABASE_URL` environment variable with Postgres connection string

4. Run migrations: `npx prisma migrate deploy`

5. Run seed script to populate production data

6. Consider replacing SHA-256 with bcrypt for production password security

7. Implement rate limiting on API routes (not included in current implementation)

8. Add environment-specific error handling and logging

## Performance Considerations

**Simulator Load**: Generating transactions every 2-5 seconds + running AI analysis is CPU-intensive; monitor server resources

**Fraud Detection Context**: `allBeneficiaries` array in analysis context is expensive (duplicate detection); consider limiting scope or implementing caching

**Socket.IO Scaling**: In-memory Socket.IO instance doesn't scale across serverless functions; for production, consider Redis adapter or separate WebSocket server

**Database Queries**: Seed script creates indexes on foreign keys; add composite indexes for frequent query patterns (e.g., transaction date + beneficiary ID)

**Real-Time Dashboard**: Limit concurrent simulator instances to 1; multiple admins starting simulators causes race conditions

## Testing Approach

No formal test suite currently exists. Recommended testing workflow:

1. **Seed data integrity**: Run seed script, verify counts match expected values
2. **Authentication flows**: Test all 5 role logins, verify dashboard access
3. **Simulator**: Run for 5+ minutes, verify transaction generation and alert creation
4. **Fraud injection**: Test all injection types, confirm alerts with correct severity
5. **Hash chain**: Audit dashboard should show "✅ Integrity Verified"
6. **Case workflow**: Assign case to inspector, add notes, submit verdict
7. **Public dashboard**: Verify no authentication required, displays aggregated data

## Known Limitations

- Single-server in-memory simulator (no distributed coordination)
- SQLite unsuitable for high-concurrency production workloads
- SHA-256 password hashing less secure than bcrypt (acceptable for demo/hackathon)
- No rate limiting or DDoS protection
- Socket.IO state lost on server restart (ephemeral connections)
- Fraud detection models not trainable (rule-based, no ML model persistence)
- No email notifications for alerts (infrastructure for future enhancement)

## Smart India Hackathon Context

This project was built for SIH 2026 demonstrating:
- Real-time fraud detection with explainable AI
- Government transparency through public dashboards
- Cryptographic audit trails for accountability
- Live digital twin simulation for demo impact
- Production-ready architecture (despite demo constraints)

**Demo Priority**: Emphasize live simulation, fraud injection, explainable alerts, and hash chain verification for maximum judge impact in 2-minute presentation window.
