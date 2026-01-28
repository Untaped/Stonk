// static/sw.js
const CACHE_NAME = 'litlguy-v1';
const urlsToCache = [
  '/',
  '/static/icon-192.png',
  '/static/icon-512.png'
  // Add other static CSS/JS files here if you have them
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});