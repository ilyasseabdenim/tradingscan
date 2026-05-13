const { Redis } = require("@upstash/redis");

function getRedis() {
  const url =
    process.env.UPSTASH_REDIS_REST_URL ||
    process.env.KV_REST_API_URL;

  const token =
    process.env.UPSTASH_REDIS_REST_TOKEN ||
    process.env.KV_REST_API_TOKEN;

  if (!url || !token) {
    throw new Error("Redis is not connected.");
  }

  return new Redis({ url, token });
}

const DECISION_ORDER = {
  "BUY CONFIRMED": 0,
  "SELL CONFIRMED": 1,
  "SELL / EXIT LONG": 2,
  "BUY / EXIT SHORT": 3,
  "TRAIL LONG": 4,
  "TRAIL SHORT": 5,
  "HOLD LONG": 6,
  "HOLD SHORT": 7,
  "GET READY TO BUY": 8,
  "GET READY TO SELL": 9,
  "LEAN BUY": 10,
  "LEAN SELL": 11,
  "WAIT": 12,
  "LOADING": 13,
  "ERROR": 14
};

module.exports = async function handler(req, res) {
  try {
    const redis = getRedis();

    const raw = await redis.hgetall("tv:latest-signals");
    const lastUpdated = await redis.get("tv:last-updated");

    const rows = Object.values(raw || {})
      .map((value) => {
        if (typeof value === "string") {
          return JSON.parse(value);
        }
        return value;
      })
      .sort((a, b) => {
        const da = DECISION_ORDER[a.decision] ?? 99;
        const db = DECISION_ORDER[b.decision] ?? 99;

        if (da !== db) return da - db;

        return (b.confidence || 0) - (a.confidence || 0);
      });

    return res.status(200).json({
      ok: true,
      total: rows.length,
      confirmed: rows.filter((r) =>
        ["BUY CONFIRMED", "SELL CONFIRMED"].includes(r.decision)
      ).length,
      active_positions: rows.filter((r) =>
        ["LONG", "SHORT"].includes(r.position)
      ).length,
      last_updated: lastUpdated,
      rows
    });

  } catch (error) {
    return res.status(500).json({
      ok: false,
      error: error.message
    });
  }
};
