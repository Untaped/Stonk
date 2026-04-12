const CACHE_NAME = 'litlguy-v1';
const urlsToCache = [
  '/',
  '/manifest.json' // Changed this to match index.html
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});