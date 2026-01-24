import './globals.css';
import { Inter, Space_Grotesk } from 'next/font/google';
import AuthProvider from '@/components/providers/auth-provider';
import { SmoothScrollProvider } from '@/components/providers/smooth-scroll';
import AICommander from '@/components/ai/ai-commander';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
});

const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  variable: '--font-space-grotesk',
});

export const metadata = {
  title: 'GrainSecure - AI-Enabled PDS Monitoring Platform',
  description: 'Enterprise-grade AI monitoring for India\'s Public Distribution System. Real-time fraud detection, network analysis, and tamper-proof accountability.',
  keywords: 'PDS, fraud detection, AI, machine learning, public distribution, government, India',
  authors: [{ name: 'GrainSecure Team' }],
  openGraph: {
    title: 'GrainSecure - AI-Enabled PDS Monitoring',
    description: 'Eliminate fraud. Protect every grain.',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${spaceGrotesk.variable} font-sans antialiased`}>
        <AuthProvider>
          <SmoothScrollProvider>
            {children}
            <AICommander />
          </SmoothScrollProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
