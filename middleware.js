export { default } from 'next-auth/middleware';

export const config = {
    matcher: [
        '/admin/:path*',
        '/inspector/:path*',
        '/dealer/:path*',
        '/auditor/:path*',
        '/beneficiary/:path*',
    ],
};
