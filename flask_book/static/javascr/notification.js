class NotificationSystem {
    constructor() {
        this.bell = document.querySelector('.notification-bell');
        this.panel = document.getElementById('notificationPanel');
        this.notificationBody = document.getElementById('notificationBody');
        this.statusDot = document.getElementById('onlineStatus');
        this.statusText = document.getElementById('statusText');
        this.count = 0;
        this.notifications = [];
        
        this.initialize();
    }

    initialize() {
        this.bell.addEventListener('click', () => {
            this.togglePanel();
        });

        document.addEventListener('click', (e) => {
            if (!this.bell.contains(e.target) && !this.panel.contains(e.target)) {
                this.panel.style.display = 'none';
            }
        });

        this.updateOnlineStatus();
        window.addEventListener('online', () => this.updateOnlineStatus());
        window.addEventListener('offline', () => this.updateOnlineStatus());

        this.checkLoginStatus();
    }

    updateOnlineStatus() {
        if (navigator.onLine) {
            this.statusDot.classList.add('status-online');
            this.statusDot.classList.remove('status-offline');
            this.statusText.textContent = 'Online';
        } else {
            this.statusDot.classList.add('status-offline');
            this.statusDot.classList.remove('status-online');
            this.statusText.textContent = 'Offline';
        }
    }

    togglePanel() {
        this.panel.style.display = this.panel.style.display === 'none' ? 'block' : 'none';
    }

    addNotification(message, type = 'info') {
        const notification = {
            id: Date.now(),
            message,
            type,
            timestamp: new Date().toLocaleString()
        };

        this.notifications.unshift(notification);
        this.updateNotificationUI();
        this.count++;
        this.updateBadge();
    }

    updateBadge() {
        const badge = document.getElementById('notificationCount');
        badge.textContent = this.count;
        badge.style.display = this.count > 0 ? 'block' : 'none';
    }

    updateNotificationUI() {
        this.notificationBody.innerHTML = this.notifications
            .map(notif => `
                <div class="notification-item">
                    <div>
                        <div class="notification-message">${notif.message}</div>
                        <small class="notification-time text-muted">${notif.timestamp}</small>
                    </div>
                </div>
            `).join('');
    }

    async checkLoginStatus() {
        const userEmail = sessionStorage.getItem('userEmail');
        if (userEmail) {
            this.addNotification(`${userEmail} đã đăng nhập thành công`);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.notificationSystem = new NotificationSystem();
});