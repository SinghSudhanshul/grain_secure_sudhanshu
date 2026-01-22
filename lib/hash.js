import crypto from 'crypto';

export function hashPassword(password) {
    return crypto.createHash('sha256').update(password).digest('hex');
}

export function verifyPassword(password, hash) {
    return hashPassword(password) === hash;
}

export function generateHash(data) {
    return crypto.createHash('sha256').update(JSON.stringify(data)).digest('hex');
}

export function generateAuditHash(prevHash, eventType, metaJson, createdAt) {
    const data = `${prevHash}${eventType}${JSON.stringify(metaJson)}${createdAt}`;
    return crypto.createHash('sha256').update(data).digest('hex');
}
