# 🚀 GrainSecure - AI-Enabled PDS Monitoring System

> **Next-generation Public Distribution System monitoring with AI-powered fraud detection, ML model auto-upgrading, and Apple-inspired UI**

[![Next.js](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)](https://www.typescriptlang.org/)
[![Prisma](https://img.shields.io/badge/Prisma-5.0-2D3748)](https://www.prisma.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## ✨ Key Features

### 🤖 AI-Powered Systems
- **Code Quality Analyzer** - Automatically detects complexity, performance issues, and applies fixes
- **ML Model Auto-Upgrader** - Monitors performance and upgrades models automatically
- **Anomaly Detection** - Isolation Forest algorithm for transaction fraud detection
- **Risk Scoring** - Weighted factor model for shop/beneficiary assessment
- **Network Analysis** - Graph algorithms for collusion pattern detection

### 🎨 Premium UI/UX
- **Apple-Inspired Design** - Glassmorphism effects with backdrop blur
- **Smooth Animations** - 200-400ms transitions with ease curves
- **Responsive Layout** - Mobile-first design for all devices
- **Interactive Charts** - Recharts, D3.js, and Mapbox visualizations
- **Real-time Updates** - Live data with React Query

### 📊 Comprehensive Dashboards
- **Administrator** - Full analytics, alerts, and system management
- **Field Inspector** - Mobile-optimized investigation tools
- **Beneficiary Portal** - Transaction verification and complaints

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- PostgreSQL 14+
- npm or yarn

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/grain_secure_sudhanshu.git
cd grain_secure_sudhanshu

# Install dependencies
npm install

# Setup environment
cp .env.example .env
# Edit .env with your database URL

# Run database migrations
npx prisma migrate dev

# Seed database
npm run seed

# Start development server
npm run dev
```

Visit `http://localhost:3000`

## 🛠️ Available Scripts

```bash
npm run dev              # Start development server
npm run build            # Build for production
npm run start            # Start production server
npm run lint             # Run ESLint
npm run seed             # Seed database with sample data

# AI Tools
npm run analyze-code     # Analyze code quality
npm run analyze-code:fix # Analyze and auto-fix issues
npm run upgrade-models   # Check and upgrade ML models
```

## 📁 Project Structure

```
grain_secure_sudhanshu/
├── app/                      # Next.js app directory
│   ├── admin/               # Admin dashboard
│   ├── inspector/           # Inspector interface
│   ├── beneficiary/         # Beneficiary portal
│   └── api/                 # API routes
├── components/
│   ├── ui/                  # Reusable UI components
│   │   ├── glass-card.tsx   # Glassmorphism cards
│   │   ├── metric-card.tsx  # Animated metrics
│   │   └── data-table.tsx   # Enhanced tables
│   └── charts/              # Data visualizations
│       ├── leakage-chart.tsx
│       └── network-graph.tsx
├── lib/
│   ├── ai/                  # AI systems
│   │   ├── code-quality-analyzer.ts
│   │   └── model-upgrader.ts
│   └── ml/                  # ML algorithms
├── prisma/
│   └── schema.prisma        # Database schema
└── scripts/                 # Utility scripts
```

## 🎯 Core Components

### AI Code Quality Analyzer

Automatically analyzes your codebase and suggests improvements:

```bash
npm run analyze-code
```

**Features:**
- Complexity detection
- Performance issue identification
- Best practice enforcement
- Auto-fixing capabilities
- HTML report generation

### ML Model Auto-Upgrader

Monitors and upgrades ML models automatically:

```bash
npm run upgrade-models
```

**Features:**
- Performance tracking
- Hyperparameter optimization
- A/B testing
- Automatic deployment
- Version management

### Premium UI Components

```tsx
import { GlassCard, MetricCard, DataTable } from '@/components/ui';

// Glassmorphism card
<GlassCard variant="elevated" hover>
  <GlassCardContent>...</GlassCardContent>
</GlassCard>

// Animated metric
<MetricCard
  title="Total Beneficiaries"
  value={850000}
  change={12.5}
  trend="up"
  gradient="blue"
/>

// Enhanced table
<DataTable
  data={data}
  columns={columns}
  selectable
  pageSize={20}
/>
```

## 📊 Database Schema

8 core tables with comprehensive relationships:

- **Users** - Authentication and roles
- **Beneficiaries** - PDS recipients
- **FairPriceShops** - Distribution outlets
- **Transactions** - Commodity distributions
- **StockMovements** - Inventory tracking
- **Alerts** - Fraud detections
- **Complaints** - Beneficiary feedback
- **NetworkRelationships** - Collusion patterns

## 🔐 Authentication

Role-based access control with 3 user types:

- **Administrator** - Full system access
- **Field Inspector** - Investigation tools
- **Beneficiary** - Personal portal

## 📈 Analytics & Reporting

- Leakage estimation with trends
- Geographic analysis with maps
- Network relationship visualization
- PDF report generation
- CSV data export

## 🎨 Design System

### Colors
- Primary: Blue-600 (#2563eb)
- Success: Green-600 (#16a34a)
- Warning: Orange-600 (#ea580c)
- Danger: Red-600 (#dc2626)

### Typography
- Font: Inter (San Francisco-inspired)
- Scale: 48px, 36px, 24px, 20px, 16px, 14px, 12px

### Spacing
- Base: 8px grid system
- Gaps: 16px, 24px, 32px, 48px

## 🚀 Deployment

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Set environment variables
vercel env add DATABASE_URL
vercel env add NEXTAUTH_SECRET
```

### Docker

```bash
# Build image
docker build -t grainsecure .

# Run container
docker run -p 3000:3000 grainsecure
```

## 🧪 Testing

```bash
# Unit tests
npm run test

# E2E tests
npm run test:e2e

# Coverage
npm run test:coverage
```

## 📝 API Documentation

API endpoints are automatically documented with TypeScript types.

### Key Endpoints

- `GET /api/beneficiaries` - List beneficiaries
- `GET /api/shops` - List fair price shops
- `GET /api/transactions` - List transactions
- `GET /api/alerts` - List alerts
- `POST /api/ml/detect` - Run anomaly detection
- `POST /api/ml/risk-score` - Calculate risk scores
- `GET /api/analytics/leakage` - Get leakage estimates

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for India's Public Distribution System
- Inspired by Apple's design philosophy
- Powered by Next.js, Prisma, and TypeScript

## 📞 Support

- **Documentation:** [docs.grainsecure.com](https://docs.grainsecure.com)
- **Issues:** [GitHub Issues](https://github.com/yourusername/grain_secure_sudhanshu/issues)
- **Email:** support@grainsecure.com

---

**Built with ❤️ for transparent and efficient food distribution**
