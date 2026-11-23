// Talks to the Flask API (/api/...). All requests stay on this computer.

export class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

export async function api(path, { method = "GET", body, form } = {}) {
  const opts = { method, headers: {} };
  if (form) opts.body = form;
  else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(`/api${path}`, opts);
  } catch {
    throw new ApiError("Cannot reach the QuizForge server. Is the start.bat window still open?", 0);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(data.error || `Request failed (HTTP ${res.status})`, res.status, data);
  return data;
}

export const get = (p) => api(p);
export const post = (p, body) => api(p, { method: "POST", body: body ?? {} });
export const put = (p, body) => api(p, { method: "PUT", body });
export const patch = (p, body) => api(p, { method: "PATCH", body });
export const del = (p) => api(p, { method: "DELETE" });

// Downloads a file from GET /api/<path>, or from POST when a body is given (exports of an unsaved draft).
export async function download(path, body) {
  const opts = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const res = await fetch(`/api${path}`, opts);
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new ApiError(data.error || `Download failed (HTTP ${res.status})`, res.status);
  }
  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition") || "";
  const name = (cd.match(/filename="([^"]+)"/) || [])[1] || "download";
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
  return name;
}

// Polls a background job until it finishes. onUpdate(snapshot) gets only new questions in snapshot.new_questions.
export function pollJob(jobId, onUpdate, { interval = 500 } = {}) {
  let since = 0;
  let stopped = false;
  let failures = 0;
  const done = new Promise((resolve, reject) => {
    const tick = async () => {
      if (stopped) return resolve(null);
      try {
        const snap = await get(`/jobs/${jobId}?since=${since}`);
        failures = 0;
        since = snap.count;
        onUpdate(snap);
        if (snap.status !== "running") return resolve(snap);
      } catch (err) {
        if (err.status === 404 || ++failures > 20) return reject(err);
      }
      setTimeout(tick, interval);
    };
    tick();
  });
  return { done, cancel: () => { stopped = true; } };
}
