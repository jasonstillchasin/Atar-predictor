// ATAR Workbook front end: edit a student's tasks, save/switch students, show the estimate.

const TASK_FIELDS = ["name", "mark", "out_of", "weight", "rank", "rank_of", "cohort_mean", "cohort_sd"];
const YEARS = ["year11", "year12"];
const SOURCE_LABEL = { entered: "your rank", task_ranks: "task ranks", z_score: "cohort stats", default: "marks only" };
const LAST_STUDENT_KEY = "atar-workbook:last-student";
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = {
  students: [],
  courses: new Map(), // UAC course name -> {units, english}
  currentId: null,
  profile: blankProfile(),
  savedSnapshot: "",
  dirty: false,
  showAdvanced: false,
  result: null,
  resultError: null,
  pending: null, // client-side problems blocking a new estimate
  predicting: false,
  predictSeq: 0,
  dataMissing: null,
  lastMap: null,
  shownAtar: null,
};

// ── Helpers ─────────────────────────────────────────────

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const str = (v) => (v === null || v === undefined ? "" : String(v));
const fmt = (v, d = 1) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));
const fmtAtar = (v) => (v <= 30 ? "≤30" : Number(v).toFixed(2));
const topPct = (p) => `Top ${Math.max(1, Math.round((1 - p) * 100))}%`;

function num(value) {
  const text = String(value ?? "").trim();
  if (text === "") return undefined;
  const n = Number(text);
  return Number.isFinite(n) ? n : NaN;
}

function blankTask() {
  return Object.fromEntries(TASK_FIELDS.map((f) => [f, ""]));
}
function blankCourse() {
  return { name: "", cohort_size: "", rank: "", year11: [blankTask()], year12: [blankTask()] };
}
function blankProfile() {
  return { name: "", cohort_strength: "", courses: [blankCourse()] };
}
const isBlankTask = (t) => TASK_FIELDS.every((f) => String(t[f]).trim() === "");

function remember(id) {
  try {
    if (id) localStorage.setItem(LAST_STUDENT_KEY, id);
  } catch { /* storage unavailable */ }
}
function recall() {
  try {
    return localStorage.getItem(LAST_STUDENT_KEY);
  } catch {
    return null;
  }
}

function relTime(iso) {
  const seconds = (new Date(iso).getTime() - Date.now()) / 1000;
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  for (const [unit, size] of [["year", 31536000], ["month", 2592000], ["day", 86400], ["hour", 3600], ["minute", 60]]) {
    if (Math.abs(seconds) >= size) return rtf.format(Math.round(seconds / size), unit);
  }
  return "just now";
}

let toastTimer;
function toast(message, kind = "") {
  const el = $("#toast");
  el.textContent = message;
  el.className = `toast ${kind}`;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), kind === "error" ? 6000 : 2200);
}

// ── API ─────────────────────────────────────────────────

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (res.status === 204) return null;
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const err = new Error(formatError(body, res.status));
    err.status = res.status;
    throw err;
  }
  return body;
}

function formatError(body, status) {
  if (!body) return `Request failed (${status})`;
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail)) {
    return body.detail.map((d) => `${describeLoc(d.loc)}${String(d.msg).replace(/^Value error, /, "")}`).join("\n");
  }
  return `Request failed (${status})`;
}

// Translate a server validation path (on the cleaned profile) back to what the user sees.
function describeLoc(loc = []) {
  const map = state.lastMap;
  const parts = [];
  let courseIdx = null;
  for (let i = 0; i < loc.length; i++) {
    const key = loc[i];
    if (key === "courses" && typeof loc[i + 1] === "number") {
      courseIdx = loc[++i];
      const original = map ? map.courses[courseIdx] : courseIdx;
      parts.push(state.profile.courses[original]?.name || `Course ${original + 1}`);
    } else if (YEARS.includes(key) && typeof loc[i + 1] === "number") {
      const j = loc[++i];
      const original = map?.tasks[`${courseIdx}-${key}`]?.[j] ?? j;
      parts.push(`Year ${key.slice(4)} task ${original + 1}`);
    } else if (typeof key === "string" && !["body", "profile"].includes(key)) {
      parts.push(key.replace(/_/g, " "));
    }
  }
  return parts.length ? `${parts.join(" › ")}: ` : "";
}

// ── Profile conversion ──────────────────────────────────

function fromApi(profile) {
  const tasks = (list) => {
    const rows = (list || []).map((t) =>
      Object.fromEntries(TASK_FIELDS.map((f) => [f, f === "out_of" && Number(t.out_of) === 100 ? "" : str(t[f])])),
    );
    return rows.length ? rows : [blankTask()];
  };
  const courses = (profile.courses || []).map((c) => ({
    name: str(c.name),
    cohort_size: str(c.cohort_size),
    rank: str(c.rank),
    year11: tasks(c.year11),
    year12: tasks(c.year12),
  }));
  return { name: str(profile.name), cohort_strength: str(profile.cohort_strength), courses: courses.length ? courses : [blankCourse()] };
}

// Editor state (strings, blank rows) -> API profile, plus a map back to editor indices.
function cleanProfile(p) {
  const map = { courses: [], tasks: {} };
  const problems = [];
  const courses = [];

  p.courses.forEach((c, ci) => {
    const allTasks = [...c.year11, ...c.year12];
    const empty = !c.name.trim() && allTasks.every(isBlankTask) && !String(c.cohort_size).trim() && !String(c.rank).trim();
    if (empty) return;

    const name = c.name.trim();
    const label = name || `Course ${ci + 1}`;
    if (!name) problems.push(`${label}: choose a course`);
    else if (state.courses.size && !state.courses.has(name)) problems.push(`${label}: not a UAC course name — pick one from the list`);

    const out = { name };
    for (const f of ["cohort_size", "rank"]) {
      const v = num(c[f]);
      if (Number.isNaN(v)) problems.push(`${label}: ${f.replace("_", " ")} must be a number`);
      else if (v !== undefined) out[f] = v;
    }
    if (out.rank !== undefined && out.cohort_size === undefined) problems.push(`${label}: add the cohort size for your rank`);

    const idx = courses.length;
    map.courses[idx] = ci;
    for (const year of YEARS) {
      out[year] = [];
      map.tasks[`${idx}-${year}`] = [];
      c[year].forEach((t, ti) => {
        if (isBlankTask(t)) return;
        const task = { name: t.name.trim() || `Task ${ti + 1}` };
        for (const f of TASK_FIELDS.slice(1)) {
          const v = num(t[f]);
          if (Number.isNaN(v)) problems.push(`${label} › ${task.name}: ${f.replace("_", " ")} must be a number`);
          else if (v !== undefined) task[f] = v;
        }
        if (task.weight === undefined) problems.push(`${label} › ${task.name}: add its weighting`);
        map.tasks[`${idx}-${year}`].push(ti);
        out[year].push(task);
      });
    }
    courses.push(out);
  });

  if (!courses.length) problems.push("Add a course and its assessment tasks");
  const profile = { name: p.name.trim() || null, cohort_strength: p.cohort_strength || null, courses };
  return { profile, map, problems };
}

function yearStats(tasks) {
  let total = 0, doneWeight = 0, sum = 0, any = false;
  for (const t of tasks) {
    if (isBlankTask(t)) continue;
    any = true;
    const w = num(t.weight);
    if (!(w > 0)) continue;
    total += w;
    const mark = num(t.mark);
    const outOf = num(t.out_of) ?? 100;
    if (mark === undefined || Number.isNaN(mark) || !(outOf > 0)) continue;
    doneWeight += w;
    sum += (mark / outOf) * 100 * w;
  }
  return { any, total, weighted: doneWeight ? sum / doneWeight : null, done: total ? doneWeight / total : 0 };
}

function unitsFor(name) {
  const info = state.courses.get(name);
  if (!info) return null;
  if (name === "Mathematics Extension 1" && state.profile.courses.some((c) => c.name.trim() === "Mathematics Extension 2")) return 2;
  return info.units;
}

// ── Editor rendering ────────────────────────────────────

function taskRow(t, ti) {
  const cell = (field, cls, placeholder, label, mode = "decimal") =>
    `<td class="${cls}"><input data-field="${field}" value="${esc(t[field])}" placeholder="${placeholder}" aria-label="${label}"${
      field === "name" ? "" : ` inputmode="${mode}"`
    } autocomplete="off"></td>`;
  return `<tr data-task="${ti}">
    ${cell("name", "c-name", "Task name", "Task name")}
    ${cell("mark", "c-num", "not sat", "Mark")}
    ${cell("out_of", "c-num", "100", "Out of")}
    ${cell("weight", "c-num", "%", "Weight %")}
    ${cell("rank", "c-num adv", "–", "Task rank", "numeric")}
    ${cell("rank_of", "c-num adv", "–", "Ranked out of", "numeric")}
    ${cell("cohort_mean", "c-num adv", "–", "Cohort mean")}
    ${cell("cohort_sd", "c-num adv", "–", "Cohort SD")}
    <td class="c-act"><button class="icon-btn" data-action="remove-task" title="Remove task" aria-label="Remove task">×</button></td>
  </tr>`;
}

function yearBlock(course, year) {
  return `<section class="year" data-year="${year}">
    <div class="year-head"><h3>Year ${year.slice(4)}</h3><div class="year-meta" data-meta></div></div>
    <div class="table-wrap"><table class="tasks">
      <thead><tr>
        <th class="c-name">Task</th><th class="c-num">Mark</th><th class="c-num">Out of</th><th class="c-num">Weight %</th>
        <th class="c-num adv">Rank</th><th class="c-num adv">of</th><th class="c-num adv">Mean</th><th class="c-num adv">SD</th><th class="c-act"></th>
      </tr></thead>
      <tbody>${course[year].map((t, ti) => taskRow(t, ti)).join("")}</tbody>
    </table></div>
    <button class="add-task" data-action="add-task">+ Add task</button>
  </section>`;
}

function courseCard(course, ci) {
  const el = document.createElement("article");
  el.className = "course";
  el.dataset.course = ci;
  el.style.setProperty("--i", ci);
  el.innerHTML = `
    <header class="course-head">
      <span class="course-index">${String(ci + 1).padStart(2, "0")}</span>
      <input class="course-name" data-field="name" list="course-options" value="${esc(course.name)}"
             placeholder="Course, e.g. Mathematics Advanced" aria-label="Course name" autocomplete="off">
      <div class="course-fields">
        <span class="units" data-units></span>
        <label class="mini mini-cohort">Cohort <input data-field="cohort_size" inputmode="numeric" value="${esc(course.cohort_size)}" placeholder="–"></label>
        <label class="mini mini-rank">Rank <input data-field="rank" inputmode="numeric" value="${esc(course.rank)}" placeholder="–" title="Your current overall rank in this course, if known"></label>
      </div>
      <button class="icon-btn" data-action="remove-course" title="Remove course" aria-label="Remove course">×</button>
    </header>
    <div class="years">${YEARS.map((y) => yearBlock(course, y)).join("")}</div>`;
  return el;
}

function renderEditor(entering = false) {
  $("#student-name").value = state.profile.name;
  $("#cohort-strength").value = state.profile.cohort_strength;
  $("#show-advanced").checked = state.showAdvanced;
  document.body.classList.toggle("show-advanced", state.showAdvanced);

  const root = $("#courses");
  root.classList.toggle("entering", entering && !REDUCED_MOTION);
  root.replaceChildren(...state.profile.courses.map(courseCard));
  state.profile.courses.forEach((_, ci) => updateCourseMeta(ci));
  renderAverages();
  if (entering) setTimeout(() => root.classList.remove("entering"), 900);
}

function updateCourseMeta(ci) {
  const card = document.querySelector(`.course[data-course="${ci}"]`);
  const course = state.profile.courses[ci];
  if (!card || !course) return;

  const name = course.name.trim();
  const units = unitsFor(name);
  const unitsEl = $("[data-units]", card);
  unitsEl.textContent = units ? `${units} unit${units > 1 ? "s" : ""}` : name && state.courses.size ? "not a UAC course" : "";
  unitsEl.classList.toggle("warn", Boolean(name) && !units && state.courses.size > 0);

  for (const year of YEARS) {
    const s = yearStats(course[year]);
    const meta = $(`.year[data-year="${year}"] [data-meta]`, card);
    if (!s.any) {
      meta.innerHTML = `<span class="muted small">No tasks yet</span>`;
      continue;
    }
    const weightOk = Math.abs(s.total - 100) <= 0.5;
    const done = Math.round(s.done * 100);
    meta.innerHTML = `
      <span class="pill ${weightOk ? "ok" : "warn"}" title="Weightings for the year should add up to 100%">Σ ${fmt(s.total, 0)}%</span>
      <span class="pill" title="Weighted mark over completed tasks">${s.weighted === null ? "no marks yet" : `${fmt(s.weighted, 1)}%`}</span>
      <span class="progress" title="${done}% of this year's weighting completed"><i style="width:${done}%"></i></span>
      <span class="muted small">${done}% done</span>`;
  }
  card.querySelectorAll("input[inputmode]").forEach((input) => input.classList.toggle("invalid", Number.isNaN(num(input.value))));
}

// ── Averages ────────────────────────────────────────────
// Mirrors atar_predictor/averages.py: a subject's overall mark weights each year by how much of
// it is complete; "All subjects" is the plain mean over subjects that have a mark.

const meanOf = (values, weights) => {
  let sum = 0, total = 0;
  values.forEach((v, i) => {
    const w = weights ? weights[i] : 1;
    if (v !== null && w > 0) {
      sum += v * w;
      total += w;
    }
  });
  return total ? sum / total : null;
};

function computeAverages() {
  const subjects = state.profile.courses
    .filter((c) => c.name.trim() || YEARS.some((y) => c[y].some((t) => !isBlankTask(t))))
    .map((c, i) => {
      const stats = Object.fromEntries(YEARS.map((y) => [y, yearStats(c[y])]));
      const marks = YEARS.map((y) => stats[y].weighted);
      const done = YEARS.map((y) => (stats[y].weighted === null ? 0 : stats[y].done));
      const name = c.name.trim();
      return {
        course: name || `Course ${i + 1}`,
        units: unitsFor(name) ?? (name.includes("Extension") ? 1 : 2),
        year11: marks[0],
        year12: marks[1],
        done11: done[0],
        done12: done[1],
        overall: meanOf(marks, done),
      };
    });
  const units = subjects.map((s) => s.units);
  const col = (key) => subjects.map((s) => s[key]);
  return {
    subjects,
    year11: meanOf(col("year11")),
    year12: meanOf(col("year12")),
    overall: meanOf(col("overall")),
    unitWeighted: { year11: meanOf(col("year11"), units), year12: meanOf(col("year12"), units), overall: meanOf(col("overall"), units) },
  };
}

function avgCell(value, doneShare) {
  if (value === null) return `<td class="avg-empty">–</td>`;
  const partial = doneShare !== undefined && doneShare < 0.995 ? `<small>${Math.round(doneShare * 100)}% done</small>` : "";
  return `<td><span class="num">${fmt(value, 1)}</span>${partial}<span class="avg-bar"><i style="width:${Math.min(100, Math.max(0, value)).toFixed(1)}%"></i></span></td>`;
}

function renderAverages() {
  const el = $("#averages");
  if (!el) return;
  const avg = computeAverages();
  const head = `<div class="report-head">
      <h2 id="averages-title">Averages</h2>
      <span class="hint">Weighted school marks · overall weights each year by how much of it is done</span>
    </div>`;
  if (!avg.subjects.length) {
    el.innerHTML = `${head}<p class="muted small">Averages appear once you add a course with marks.</p>`;
    return;
  }
  const count = (key) => avg.subjects.filter((s) => s[key] !== null).length;
  const rows = avg.subjects
    .map((s) => `<tr><th scope="row">${esc(s.course)}</th>${avgCell(s.year11, s.done11)}${avgCell(s.year12, s.done12)}${avgCell(s.overall)}</tr>`)
    .join("");
  const footCell = (key) =>
    avg[key] === null
      ? `<td class="avg-empty">–</td>`
      : `<td title="Weighted by units: ${fmt(avg.unitWeighted[key], 1)}"><span class="num">${fmt(avg[key], 1)}</span><small>${count(key)} of ${avg.subjects.length} subjects</small></td>`;
  el.innerHTML = `${head}
    <div class="table-wrap"><table class="avg-table">
      <thead><tr><th>Subject</th><th>Year 11</th><th>Year 12</th><th>Overall</th></tr></thead>
      <tbody>${rows}</tbody>
      <tfoot><tr><th scope="row">All subjects</th>${footCell("year11")}${footCell("year12")}${footCell("overall")}</tr></tfoot>
    </table></div>`;
}

// ── Students ────────────────────────────────────────────

function renderStudents() {
  const query = $("#student-search").value.trim().toLowerCase();
  const matches = state.students.filter((s) => !query || String(s.name).toLowerCase().includes(query));
  const unsaved = state.currentId
    ? ""
    : `<button class="student active unsaved" disabled><span class="s-name">${esc(state.profile.name || "New student")}</span><span class="s-meta">not saved yet</span></button>`;
  const rows = matches.map(
    (s) => `<button class="student ${s.id === state.currentId ? "active" : ""}" data-id="${esc(s.id)}">
      <span class="s-name">${esc(s.name)}</span>
      <span class="s-meta">${s.courses} course${s.courses === 1 ? "" : "s"} · ${relTime(s.updated)}</span>
    </button>`,
  );
  const empty = rows.length ? "" : `<p class="muted small">${state.students.length ? "No matches" : "No saved students yet"}</p>`;
  $("#student-list").innerHTML = unsaved + rows.join("") + empty;
}

async function refreshStudents() {
  state.students = await api("/api/students");
  renderStudents();
}

function setProfile(profile, id) {
  state.profile = profile;
  state.currentId = id;
  state.savedSnapshot = JSON.stringify(profile);
  state.dirty = false;
  state.showAdvanced = profile.courses.some(
    (c) => YEARS.some((y) => c[y].some((t) => ["rank", "rank_of", "cohort_mean", "cohort_sd"].some((f) => String(t[f]).trim()))),
  );
  state.result = null;
  state.resultError = null;
  state.pending = null;
  state.shownAtar = null;
  remember(id);
  renderStudents();
  renderEditor(true);
  updateSaveStatus();
  renderResults();
  runPredict();
}

const confirmDiscard = () => !state.dirty || confirm("Discard unsaved changes to this student?");

async function selectStudent(id) {
  if (id === state.currentId || !confirmDiscard()) return;
  try {
    const res = await api(`/api/students/${encodeURIComponent(id)}`);
    setProfile(fromApi(res.profile), id);
  } catch (err) {
    toast(err.message, "error");
  }
}

function newStudent() {
  if (!confirmDiscard()) return;
  setProfile(blankProfile(), null);
  $("#student-name").focus();
}

function duplicateStudent() {
  const copy = structuredClone(state.profile);
  copy.name = `${copy.name.trim() || "Student"} (copy)`;
  state.profile = copy;
  state.currentId = null;
  state.savedSnapshot = "";
  renderEditor();
  changed({ predict: false });
  renderStudents();
  toast("Duplicated — save to keep the copy");
}

async function deleteStudent() {
  if (!state.currentId) return;
  const name = state.profile.name || "this student";
  if (!confirm(`Delete ${name}? This removes their saved file.`)) return;
  try {
    await api(`/api/students/${encodeURIComponent(state.currentId)}`, { method: "DELETE" });
    state.dirty = false;
    await refreshStudents();
    if (state.students.length) {
      state.currentId = null;
      await selectStudent(state.students[0].id);
    } else {
      setProfile(blankProfile(), null);
    }
    toast(`Deleted ${name}`);
  } catch (err) {
    toast(err.message, "error");
  }
}

async function saveStudent() {
  const { profile, map, problems } = cleanProfile(state.profile);
  if (!profile.name) {
    toast("Give the student a name before saving", "error");
    $("#student-name").focus();
    return;
  }
  if (problems.length) {
    toast(`Can’t save yet:\n${problems.slice(0, 4).join("\n")}`, "error");
    return;
  }
  state.lastMap = map;
  updateSaveStatus("Saving…");
  try {
    if (state.currentId) {
      await api(`/api/students/${encodeURIComponent(state.currentId)}`, { method: "PUT", body: JSON.stringify(profile) });
    } else {
      const res = await api("/api/students", { method: "POST", body: JSON.stringify(profile) });
      state.currentId = res.id;
      remember(res.id);
    }
    state.savedSnapshot = JSON.stringify(state.profile);
    state.dirty = false;
    await refreshStudents();
    updateSaveStatus();
    toast(`Saved ${profile.name}`);
  } catch (err) {
    updateSaveStatus();
    toast(err.message, "error");
  }
}

function updateSaveStatus(message) {
  const el = $("#save-status");
  if (message) {
    el.textContent = message;
    el.className = "save-status";
    return;
  }
  el.textContent = state.dirty ? "Unsaved changes" : state.currentId ? "Saved" : "Not saved yet";
  el.className = `save-status ${state.dirty ? "dirty" : state.currentId ? "clean" : ""}`;
  $("#delete-student").disabled = !state.currentId;
}

// ── Change handling ─────────────────────────────────────

let predictTimer;
function changed({ predict = true } = {}) {
  state.dirty = JSON.stringify(state.profile) !== state.savedSnapshot;
  updateSaveStatus();
  renderAverages();
  if (!state.currentId) renderStudents();
  if (predict) {
    clearTimeout(predictTimer);
    predictTimer = setTimeout(runPredict, 550);
  }
}

function bindEditorEvents() {
  const root = $("#courses");

  root.addEventListener("input", (event) => {
    const input = event.target.closest("input[data-field]");
    if (!input) return;
    const card = input.closest(".course");
    const ci = Number(card.dataset.course);
    const course = state.profile.courses[ci];
    const row = input.closest("tr[data-task]");
    if (row) {
      const year = input.closest(".year").dataset.year;
      course[year][Number(row.dataset.task)][input.dataset.field] = input.value;
      updateCourseMeta(ci);
    } else {
      course[input.dataset.field] = input.value;
      state.profile.courses.forEach((_, i) => updateCourseMeta(i)); // Maths Ext 1 units depend on Ext 2
    }
    changed();
  });

  root.addEventListener("click", (event) => {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const card = button.closest(".course");
    const ci = Number(card.dataset.course);
    const course = state.profile.courses[ci];
    const year = button.closest(".year")?.dataset.year;

    if (button.dataset.action === "remove-course") {
      const hasContent = course.name.trim() || YEARS.some((y) => course[y].some((t) => !isBlankTask(t)));
      if (hasContent && !confirm(`Remove ${course.name.trim() || `course ${ci + 1}`} and its tasks?`)) return;
      state.profile.courses.splice(ci, 1);
      if (!state.profile.courses.length) state.profile.courses.push(blankCourse());
      renderEditor();
    } else if (button.dataset.action === "add-task") {
      course[year].push(blankTask());
      renderEditor();
      focusTask(ci, year, course[year].length - 1);
    } else if (button.dataset.action === "remove-task") {
      course[year].splice(Number(button.closest("tr").dataset.task), 1);
      if (!course[year].length) course[year].push(blankTask());
      renderEditor();
    }
    changed();
  });

  // Enter in the last task row adds another row.
  root.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    const row = event.target.closest("tr[data-task]");
    if (!row) return;
    event.preventDefault();
    const ci = Number(row.closest(".course").dataset.course);
    const year = row.closest(".year").dataset.year;
    const ti = Number(row.dataset.task);
    const tasks = state.profile.courses[ci][year];
    if (ti === tasks.length - 1) {
      tasks.push(blankTask());
      renderEditor();
      changed({ predict: false });
    }
    focusTask(ci, year, ti + 1);
  });
}

function focusTask(ci, year, ti) {
  $(`.course[data-course="${ci}"] .year[data-year="${year}"] tr[data-task="${ti}"] input[data-field="name"]`)?.focus();
}

// ── Prediction & results ────────────────────────────────

async function runPredict() {
  const { profile, map, problems } = cleanProfile(state.profile);
  if (problems.length) {
    state.pending = problems;
    renderResults();
    return;
  }
  state.pending = null;
  const seq = ++state.predictSeq;
  state.predicting = true;
  $("#results .panel")?.classList.add("busy");
  try {
    state.lastMap = map;
    const result = await api("/api/predict", { method: "POST", body: JSON.stringify({ profile }) });
    if (seq !== state.predictSeq) return;
    state.result = result;
    state.resultError = null;
  } catch (err) {
    if (seq !== state.predictSeq) return;
    if (err.status === 503) state.dataMissing = err.message;
    state.resultError = err.message;
  } finally {
    if (seq === state.predictSeq) {
      state.predicting = false;
      renderResults();
    }
  }
}

const CIRCLE_SVG = `<svg class="circle" viewBox="0 0 240 110" preserveAspectRatio="none" aria-hidden="true">
  <path d="M40 22 C 110 -2, 215 6, 232 48 C 246 86, 160 108, 90 102 C 30 97, 2 74, 10 48 C 16 28, 52 14, 96 10"/>
</svg>`;

function notesHtml(notes) {
  if (!notes.length) return "";
  return `<ul class="notes">${notes.map(([kind, text]) => `<li class="${kind}">${esc(text)}</li>`).join("")}</ul>`;
}

function distributionSvg(r) {
  if (!r.histogram) return "";
  const W = 380, H = 124, pad = { l: 10, r: 10, t: 12, b: 24 };
  const { edges, counts } = r.histogram;
  let lo = Math.max(0, Math.floor(Math.min(edges[0], r.atar.p10)) - 1);
  let hi = Math.min(100, Math.ceil(Math.max(edges.at(-1), r.atar.p90)) + 1);
  if (hi - lo < 8) {
    const mid = (hi + lo) / 2;
    lo = Math.max(0, mid - 4);
    hi = Math.min(100, lo + 8);
  }
  const x = (v) => pad.l + ((v - lo) / (hi - lo)) * (W - pad.l - pad.r);
  const base = H - pad.b;
  const maxCount = Math.max(...counts, 1);
  const bars = counts
    .map((c, i) => {
      if (!c) return "";
      const h = (c / maxCount) * (base - pad.t);
      const w = Math.max(1, x(edges[i + 1]) - x(edges[i]) - 1);
      return `<rect class="bar" x="${x(edges[i]).toFixed(1)}" y="${(base - h).toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}"/>`;
    })
    .join("");
  const span = hi - lo;
  const step = span > 30 ? 10 : span > 12 ? 5 : 2;
  let ticks = "";
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    ticks += `<line class="tick" x1="${x(v)}" x2="${x(v)}" y1="${base}" y2="${base + 4}"/><text x="${x(v)}" y="${H - 6}" text-anchor="middle">${v}</text>`;
  }
  const band = `<rect class="band" x="${x(r.atar.p10)}" y="${pad.t - 6}" width="${Math.max(2, x(r.atar.p90) - x(r.atar.p10))}" height="${base - pad.t + 6}"/>`;
  const median = `<line class="median" x1="${x(r.atar.p50)}" x2="${x(r.atar.p50)}" y1="${pad.t - 8}" y2="${base}"/>`;
  return `<figure class="dist">
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Simulated ATARs: 80% between ${fmtAtar(r.atar.p10)} and ${fmtAtar(r.atar.p90)}">
      ${band}${bars}${median}<line class="axis" x1="${pad.l}" x2="${W - pad.r}" y1="${base}" y2="${base}"/>${ticks}
    </svg>
    <figcaption>${r.draws.toLocaleString()} simulated outcomes · shaded = middle 80%</figcaption>
  </figure>`;
}

function rangeBar(s) {
  const p = (v) => Math.min(100, Math.max(0, (v / 50) * 100));
  return `<span class="range"><b style="left:${p(s.p10).toFixed(1)}%;width:${(p(s.p90) - p(s.p10)).toFixed(1)}%"></b><i style="left:${p(s.p50).toFixed(1)}%"></i></span>`;
}

function courseTable(r) {
  const rows = r.courses.map((c) => {
    const dots = Array.from({ length: c.units }, (_, i) => `<i class="${i < c.units_counted_at_median ? "on" : ""}"></i>`).join("");
    return `<tr>
      <th scope="row"><span class="cname">${esc(c.course)}</span>
        <span class="dots" title="${c.units_counted_at_median} of ${c.units} unit(s) count toward the aggregate">${dots}</span></th>
      <td><span class="num">${topPct(c.school_percentile)}</span>
        <small class="${c.rank_source === "default" ? "warn" : ""}">from ${SOURCE_LABEL[c.rank_source]}</small></td>
      <td><span class="num">${c.hsc_mark_p50}</span><small>${esc(c.band_p50)}</small></td>
      <td class="scaled"><span class="num">${fmt(c.scaled_per_unit.p50, 1)}</span>${rangeBar(c.scaled_per_unit)}</td>
    </tr>`;
  });
  return `<div class="table-wrap"><table class="result-table">
    <thead><tr>
      <th title="Filled dots = units that count toward the best-10 aggregate">Course</th>
      <th title="Estimated final position in the school cohort">School</th>
      <th title="Predicted HSC mark and band (median)">HSC</th>
      <th title="Scaled mark per unit out of 50: median, with the 10th–90th percentile range">Scaled</th>
    </tr></thead>
    <tbody>${rows.join("")}</tbody>
  </table></div>`;
}

function renderResults() {
  const el = $("#results");
  if (state.dataMissing) {
    el.innerHTML = `<div class="panel">
      <p class="eyebrow">Estimated ATAR</p><div class="grade-empty">—</div>
      <h2 class="section-title">Model data not found</h2>
      <p>Run these once in the project folder, then restart <code>atar-web</code>:</p>
      <pre class="setup">uv run python -m atar_predictor.data.fetch\nuv run python -m atar_predictor.data.extract</pre>
    </div>`;
    return;
  }

  const r = state.result;
  const notes = [];
  if (state.pending) notes.push(...state.pending.map((p) => ["todo", p]));
  if (state.resultError) notes.push(...state.resultError.split("\n").map((p) => ["error", p]));

  if (!r) {
    el.innerHTML = `<div class="panel ${state.predicting ? "busy" : ""}">
      <span class="busy-label">Calculating…</span>
      <p class="eyebrow">Estimated ATAR</p><div class="grade-empty">—</div>
      <p class="muted">Add courses with their assessment marks and weightings to see an estimate.</p>
      ${notesHtml(notes)}
    </div>`;
    return;
  }

  notes.push(...r.warnings.map((w) => ["warn", w]));
  const stale = Boolean(state.pending || state.resultError);
  const previous = state.shownAtar;
  el.innerHTML = `<div class="panel ${stale ? "stale" : ""} ${state.predicting ? "busy" : ""}">
    <span class="busy-label">Recalculating…</span>
    <div class="grade-row">
      <div>
        <p class="eyebrow">Estimated ATAR</p>
        <div class="grade"><span class="grade-num">${fmtAtar(r.atar.p50)}</span>${CIRCLE_SVG}</div>
      </div>
      <dl class="grade-facts">
        <div><dt>Likely range</dt><dd>${fmtAtar(r.atar.p10)} – ${fmtAtar(r.atar.p90)}</dd></div>
        <div><dt>Aggregate</dt><dd>${fmt(r.aggregate.p50, 1)}<small> / 500</small></dd></div>
        <div><dt>Scaling data</dt><dd>${r.scaling_years[0]}–${r.scaling_years.at(-1)}</dd></div>
      </dl>
    </div>
    ${distributionSvg(r)}
    ${r.eligible ? "" : notesHtml([["error", `Not ATAR-eligible: ${r.eligibility_reasons.join("; ")}`]])}
    <h3 class="section-title">Course by course</h3>
    ${courseTable(r)}
    ${notesHtml(notes)}
    <p class="fineprint">Simulated from UAC’s published scaling statistics under stated assumptions
      (school strength, how rank maps to state position). An estimate, not an official ATAR.</p>
  </div>`;

  if (previous !== null && previous !== r.atar.p50) {
    if (!REDUCED_MOTION) $(".grade .circle", el)?.classList.add("draw");
    countUp($(".grade-num", el), previous, r.atar.p50);
  }
  state.shownAtar = r.atar.p50;
}

function countUp(node, from, to) {
  if (!node || REDUCED_MOTION || from === null || from === undefined) return;
  const start = performance.now();
  const duration = 450;
  const tick = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    node.textContent = fmtAtar(from + (to - from) * eased);
    if (t < 1) requestAnimationFrame(tick);
    else node.textContent = fmtAtar(to);
  };
  requestAnimationFrame(tick);
}

// ── Init ────────────────────────────────────────────────

function bindChromeEvents() {
  $("#student-name").addEventListener("input", (e) => {
    state.profile.name = e.target.value;
    changed({ predict: false });
  });
  $("#cohort-strength").addEventListener("change", (e) => {
    state.profile.cohort_strength = e.target.value;
    changed();
  });
  $("#show-advanced").addEventListener("change", (e) => {
    state.showAdvanced = e.target.checked;
    document.body.classList.toggle("show-advanced", state.showAdvanced);
  });
  $("#add-course").addEventListener("click", () => {
    state.profile.courses.push(blankCourse());
    renderEditor();
    $(`.course[data-course="${state.profile.courses.length - 1}"] .course-name`)?.focus();
    changed({ predict: false });
  });
  $("#new-student").addEventListener("click", newStudent);
  $("#duplicate-student").addEventListener("click", duplicateStudent);
  $("#delete-student").addEventListener("click", deleteStudent);
  $("#save-student").addEventListener("click", saveStudent);
  $("#student-search").addEventListener("input", renderStudents);
  $("#student-list").addEventListener("click", (e) => {
    const button = e.target.closest("[data-id]");
    if (button) selectStudent(button.dataset.id);
  });
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      saveStudent();
    }
  });
  window.addEventListener("beforeunload", (e) => {
    if (state.dirty) e.preventDefault();
  });
}

async function init() {
  bindChromeEvents();
  bindEditorEvents();
  renderResults();

  try {
    const courses = await api("/api/courses");
    state.courses = new Map(courses.map((c) => [c.name, c]));
    $("#course-options").innerHTML = courses.map((c) => `<option value="${esc(c.name)}"></option>`).join("");
  } catch (err) {
    if (err.status === 503) state.dataMissing = err.message;
  }

  try {
    await refreshStudents();
  } catch (err) {
    toast(err.message, "error");
  }
  const last = recall();
  const target = state.students.find((s) => s.id === last) || state.students[0];
  if (target) await selectStudent(target.id);
  else setProfile(blankProfile(), null);
}

init();
