// Navigation helper shared by the pages (kept apart from app.js to avoid an import cycle).
export function go(path) {
  if (location.hash === `#/${path}`) window.dispatchEvent(new HashChangeEvent("hashchange"));
  else location.hash = `#/${path}`;
}

// Passing data to the next page without putting it in the URL (e.g. "take this quiz now").
const handoff = new Map();
export function handOff(key, value) { handoff.set(key, value); }
export function takeHandOff(key) {
  const v = handoff.get(key);
  handoff.delete(key);
  return v;
}
