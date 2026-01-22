import { Server } from 'socket.io';

let io;

export default function SocketHandler(req, res) {
    if (res.socket.server.io) {
        res.end();
        return;
    }

    io = new Server(res.socket.server);
    res.socket.server.io = io;

    io.on('connection', (socket) => {
        console.log('Client connected');

        socket.on('disconnect', () => {
            console.log('Client disconnected');
        });
    });

    res.end();
}

export function broadcastTransaction(data) {
    if (io) {
        io.emit('transaction', data);
    }
}

export function broadcastAlert(data) {
    if (io) {
        io.emit('alert', data);
    }
}

export function broadcastSimulatorStatus(data) {
    if (io) {
        io.emit('simulatorStatus', data);
    }
}

export { io };
