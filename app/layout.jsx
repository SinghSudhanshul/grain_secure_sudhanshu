import './globals.css';
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'] });

export const metadata = {
    title: 'GrainSecure - AI-Enabled PDS Monitoring',
    description: 'Intelligent Public Distribution System monitoring with AI fraud detection',
};

export default function RootLayout({ children }) {
    return (
        <html lang="en">
            <body className={inter.className}>{children}</body>
        </html>
    );
}
