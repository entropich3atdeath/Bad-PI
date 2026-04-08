-- ── Workers ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS workers (
    worker_id        TEXT PRIMARY KEY,
    gpu_type         TEXT NOT NULL,
    baseline_bpb     REAL NOT NULL,
    contact          TEXT,
    auth_token       TEXT,
    registered_at    REAL NOT NULL,
    last_seen        REAL NOT NULL,
    experiment_count INTEGER DEFAULT 0
);

-- ── Experiments ──────────────────────────────────────────────────────────────
-- One row per completed (or failed) 5-min run.
CREATE TABLE IF NOT EXISTS experiments (
    exp_id           TEXT PRIMARY KEY,
    worker_id        TEXT NOT NULL,
    config           TEXT NOT NULL,       -- JSON
    config_delta     TEXT NOT NULL,       -- JSON (subset that changed)
    val_bpb          REAL,
    delta_bpb        REAL,               -- val_bpb - worker's baseline (neg = good)
    duration_seconds REAL,
    status           TEXT DEFAULT 'completed',   -- completed | failed
    error            TEXT,
    started_at       REAL,
    completed_at     REAL NOT NULL,
    FOREIGN KEY (worker_id) REFERENCES workers(worker_id)
);

CREATE INDEX IF NOT EXISTS idx_exp_delta ON experiments(delta_bpb);
CREATE INDEX IF NOT EXISTS idx_exp_worker ON experiments(worker_id);
CREATE INDEX IF NOT EXISTS idx_exp_completed ON experiments(completed_at);

-- ── Search space dimensions ───────────────────────────────────────────────────
-- Meta-agent maintains one row per tunable hyperparameter.
CREATE TABLE IF NOT EXISTS dimensions (
    name             TEXT PRIMARY KEY,
    dtype            TEXT NOT NULL,       -- 'float_log' | 'float' | 'int' | 'categorical'
    min_val          REAL,                -- for float/int
    max_val          REAL,
    log_scale        INTEGER DEFAULT 0,
    categories       TEXT,               -- JSON array, for categorical
    frozen           INTEGER DEFAULT 0,
    frozen_value     TEXT,               -- JSON-encoded frozen value
    importance       REAL DEFAULT 0.5,   -- fANOVA score, 0-1
    n_samples        INTEGER DEFAULT 0,  -- how many experiments touched this dim
    updated_at       REAL,
    is_canary        INTEGER DEFAULT 0,
    canary_prob      REAL DEFAULT 1.0
);

-- ── Suggested config queue ────────────────────────────────────────────────────
-- Meta-agent pre-generates batches of configs; workers pop from this queue.
CREATE TABLE IF NOT EXISTS config_queue (
    exp_id           TEXT PRIMARY KEY,
    config_delta     TEXT NOT NULL,      -- JSON
    priority         REAL NOT NULL,
    note             TEXT DEFAULT '',
    assigned_to      TEXT,               -- worker_id or NULL
    assigned_at      REAL,
    status           TEXT DEFAULT 'pending'  -- pending | assigned | completed | expired
);

CREATE INDEX IF NOT EXISTS idx_queue_status ON config_queue(status, priority DESC);

-- ── Dimension interaction cache ───────────────────────────────────────────────
-- Stores pairwise fANOVA interaction scores between dimensions.
CREATE TABLE IF NOT EXISTS dim_interactions (
    dim_a            TEXT NOT NULL,
    dim_b            TEXT NOT NULL,
    interaction_score REAL NOT NULL,
    computed_at      REAL NOT NULL,
    PRIMARY KEY (dim_a, dim_b)
);

-- ── Program snapshots ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS program_snapshots (
    snapshot_id      TEXT PRIMARY KEY,
    content          TEXT NOT NULL,
    experiment_count INTEGER NOT NULL,
    created_at       REAL NOT NULL
);

-- ── Seed initial search space dimensions (autoresearch/nanochat defaults) ─────
INSERT OR IGNORE INTO dimensions
    (name, dtype, min_val, max_val, log_scale, categories, frozen, frozen_value,
     importance, n_samples, updated_at, is_canary, canary_prob)
VALUES
    ('DEPTH',             'int',         4,    24,   0, NULL,               0, NULL, 0.5, 0, 0, 0, 1.0),
    ('learning_rate',     'float_log',   1e-4, 3e-2, 1, NULL,               0, NULL, 0.5, 0, 0, 0, 1.0),
    ('TOTAL_BATCH_SIZE',  'categorical', NULL, NULL, 0, '[16384,32768,65536,131072]', 0, NULL, 0.5, 0, 0, 0, 1.0),
    ('DEVICE_BATCH_SIZE', 'int',         4,    64,   0, NULL,               0, NULL, 0.5, 0, 0, 0, 1.0),
    ('WINDOW_PATTERN',    'categorical', NULL, NULL, 0, '["L","SL","SSL","SSSL"]',   0, NULL, 0.5, 0, 0, 0, 1.0),
    ('head_dim',          'categorical', NULL, NULL, 0, '[64,128]',         0, NULL, 0.5, 0, 0, 0, 1.0),
    ('weight_decay',      'float_log',   1e-4, 1e-1, 1, NULL,               0, NULL, 0.5, 0, 0, 0, 1.0),
    ('muon_lr',           'float_log',   1e-4, 1e-2, 1, NULL,               0, NULL, 0.5, 0, 0, 0, 1.0);
