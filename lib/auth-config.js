import CredentialsProvider from 'next-auth/providers/credentials';

export const authOptions = {
  providers: [
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null;
        }

        // For demo purposes - replace with actual database check
        // const { prisma } = await import('@/lib/prisma');
        // const { verifyPassword } = await import('@/lib/hash');

        // Mock users for development
        const mockUsers = {
          'admin@grainsecure.com': { id: '1', name: 'Admin User', role: 'ADMIN', password: 'admin123' },
          'inspector@grainsecure.com': { id: '2', name: 'Inspector', role: 'INSPECTOR', password: 'inspector123' },
          'dealer@grainsecure.com': { id: '3', name: 'Dealer', role: 'DEALER', password: 'dealer123' },
          'auditor@grainsecure.com': { id: '4', name: 'Auditor', role: 'AUDITOR', password: 'auditor123' },
          'beneficiary@grainsecure.com': { id: '5', name: 'Beneficiary', role: 'BENEFICIARY', password: 'beneficiary123' },
        };

        const user = mockUsers[credentials.email];
        if (user && user.password === credentials.password) {
          return {
            id: user.id,
            email: credentials.email,
            name: user.name,
            role: user.role,
          };
        }

        return null;
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.role = user.role;
        token.id = user.id;
      }
      return token;
    },
    async session({ session, token }) {
      if (session?.user) {
        session.user.role = token.role;
        session.user.id = token.id;
      }
      return session;
    },
  },
  pages: {
    signIn: '/login',
  },
  session: {
    strategy: 'jwt',
  },
  secret: process.env.NEXTAUTH_SECRET || 'grainsecure-hackathon-secret-2026',
};
