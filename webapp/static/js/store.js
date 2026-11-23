// App-wide state shared between pages: the model status and the settings limits.
import { get } from "./api.js";

const listeners = new Set();
export const store = { status: null, settings: null };

export function onStatus(fn) {
  listeners.add(fn);
  if (store.status) fn(store.status);
  return () => listeners.delete(fn);
}

export async function refreshStatus(force = false) {
  try {
    store.status = await get(`/status${force ? "?force=1" : ""}`);
  } catch (err) {
    store.status = { connected: false, ready: false, error: err.message, models: [], installed_models: [], server_down: err.status === 0 };
  }
  listeners.forEach((fn) => fn(store.status));
  return store.status;
}

export async function loadSettings(force = false) {
  if (!store.settings || force) store.settings = await get("/settings");
  return store.settings;
}

// friendly name of the active model, never the tag (tags are shown on the Settings page only)
export function activeModelName() {
  return store.status?.model_name || "the local model";
}
