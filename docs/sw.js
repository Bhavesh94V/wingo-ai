// WinGo AI PWA Service Worker
const CACHE = 'wingo-pwa-v2';

// Only cache the main page (use correct GitHub Pages path)
self.addEventListener('install', e => {
  self.skipWaiting();
  // Don't addAll to avoid cache failures blocking install
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
});

// Network-first: always try network, fall back to cache
self.addEventListener('fetch', e => {
  // Only cache same-origin HTML/CSS/JS, not Supabase/API calls
  const url = new URL(e.request.url);
  const isSameOrigin = url.origin === self.location.origin;
  const isHTML = e.request.destination === 'document';

  if (isSameOrigin && isHTML) {
    e.respondWith(
      fetch(e.request)
        .then(res => {
          const clone = res.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
          return res;
        })
        .catch(() => caches.match(e.request))
    );
  }
  // All other requests (Supabase, fonts, etc.) go directly to network
});
